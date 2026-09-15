"""Pure, inexpensive image and pose metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

from pdc.pose_estimator import PoseKeypoint

FACE = {"nose", "left_eye", "right_eye", "left_face", "right_face"}
UPPER = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"}
LOWER = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
LEG_CHAINS = {
    "left": ("left_hip", "left_knee", "left_ankle"),
    "right": ("right_hip", "right_knee", "right_ankle"),
}

BODY_VISIBILITY_CLASSES = ("face_only", "upper_only", "lower_only", "full", "body_only")


@dataclass(frozen=True)
class BodyVisibilityConfig:
    """Conservative, model-independent thresholds for visibility post-processing."""
    keypoint_visibility: float = 0.5
    face_score: float = 0.52
    upper_score: float = 0.56
    lower_score: float = 0.56
    full_score: float = 0.68
    boundary_margin: float = 0.025
    min_person_size: int = 24
    debounce_frames: int = 3
    # Anatomical-plausibility check: two adjacent leg joints (hip-knee, knee-ankle,
    # or hip-ankle) that are both individually "confident" yet closer together than
    # this fraction of the person's own box height are almost certainly a pose-model
    # collapse under occlusion, not a real detection -- the further joint is
    # discounted (visibility zeroed) before scoring. Never applied to FACE, where
    # closely-spaced points (eyes/nose) are normal.
    min_limb_separation: float = 0.05
    # Box aspect ratio (height/width) is a cheap, keypoint-independent prior for
    # whether legs are even in frame at all. Below aspect_full_low the crop can't
    # plausibly contain visible legs; at/above aspect_full_high it strongly looks
    # like a full standing body. Blended into the "lower" score with this weight,
    # keeping keypoint evidence dominant while letting box shape correct it.
    aspect_full_low: float = 2.0
    aspect_full_high: float = 3.2
    lower_aspect_weight: float = 0.3
    # Full-body evidence is rejected when the person box is clearly torso-shaped.
    # This prevents confident but extrapolated leg points from turning upper-body
    # crops into full-body predictions.
    full_min_aspect_score: float = 0.25


def _discount_collapsed_leg_keypoints(names: np.ndarray, xy: np.ndarray, visibility: np.ndarray,
                                       box: tuple[int, int, int, int], config: BodyVisibilityConfig) -> np.ndarray:
    """Zero out knee/ankle keypoints that are implausibly close to a neighboring
    leg joint, scaled to this person's own box height. A hip, knee, and ankle on
    one leg are never all bunched together in a real human body; when a pose
    model can't see the leg it tends to collapse those predictions onto (or near)
    the hip instead of abstaining, so a tight cluster is a hallucination signal,
    not evidence of visibility.
    """
    adjusted = visibility.copy()
    box_height = max(1.0, box[3] - box[1])
    name_to_index = {name: index for index, name in enumerate(names)}

    def confident(index: int) -> bool:
        return adjusted[index] >= config.keypoint_visibility

    def too_close(index_a: int, index_b: int) -> bool:
        return float(np.linalg.norm(xy[index_a] - xy[index_b])) / box_height < config.min_limb_separation

    for hip_name, knee_name, ankle_name in LEG_CHAINS.values():
        hip_i, knee_i, ankle_i = name_to_index[hip_name], name_to_index[knee_name], name_to_index[ankle_name]
        if confident(hip_i) and confident(knee_i) and too_close(hip_i, knee_i):
            adjusted[knee_i] = 0.0
        if confident(knee_i) and confident(ankle_i) and too_close(knee_i, ankle_i):
            adjusted[ankle_i] = 0.0
        if confident(hip_i) and confident(ankle_i) and too_close(hip_i, ankle_i):
            adjusted[ankle_i] = 0.0
    return adjusted


def _aspect_full_score(box: tuple[int, int, int, int], config: BodyVisibilityConfig) -> float:
    """0..1 prior for "does this crop's shape look like a full standing body",
    from box aspect ratio alone -- independent of any keypoint confidence."""
    x1, y1, x2, y2 = box
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = height / width
    span = max(config.aspect_full_high - config.aspect_full_low, 1e-6)
    return float(np.clip((aspect - config.aspect_full_low) / span, 0.0, 1.0))


def _group_score(names: np.ndarray, xy: np.ndarray, visibility: np.ndarray, group: set[str],
                box: tuple[int, int, int, int], image_shape: tuple[int, int], config: BodyVisibilityConfig) -> tuple[float, dict]:
    """Score a region using confidence, geometry, and boundary/truncation evidence."""
    mask = np.array([name in group for name in names], dtype=bool)
    if not mask.any():
        return 0.0, {"visible": 0, "total": 0, "boundary_penalty": 0.0}
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    h, w = image_shape[:2]
    points = xy[mask]
    conf = np.clip(visibility[mask], 0.0, 1.0)
    inside = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    frame_margin = max(1.0, config.boundary_margin * min(w, h))
    near_frame = (points[:, 0] <= frame_margin) | (points[:, 1] <= frame_margin) | \
                 (points[:, 0] >= w - frame_margin) | (points[:, 1] >= h - frame_margin)
    usable = (conf >= config.keypoint_visibility) & inside
    # A high-confidence point outside the person box is an extrapolation signal.
    visible_count = int(usable.sum())
    confidence_score = float(np.mean(conf[usable])) if visible_count else 0.0
    # Region sizes differ (face has 5 points, upper 8, lower 4); score
    # confidence independently, while requiring at least two landmarks below.
    score = confidence_score * min(1.0, visible_count / 2.0)
    score *= 1.0 - 0.35 * float(np.mean(near_frame & usable))
    return score, {"visible": int(usable.sum()), "total": int(mask.sum()),
                   "boundary_penalty": round(float(np.mean(near_frame & usable)), 3)}


def classify_body_visibility(keypoints: Iterable[PoseKeypoint], person_box: tuple[int, int, int, int],
                             image_shape: tuple[int, int], pose_confidence: float = 1.0,
                             config: BodyVisibilityConfig = BodyVisibilityConfig()) -> dict:
    """Classify a pose without changing the pose model.

    Scores are intentionally conservative: `upper_score`/`lower_score` are the bar
    for a region to count as visible at all, while the stricter `full_score` bar is
    used only to flag a "full" call as high- vs. moderate-confidence via `ambiguous`
    -- it must never cause "full" evidence to be relabeled as a different class
    (e.g. "body_only"), since that would throw away real detections rather than
    just annotate their confidence.

    Two extra, keypoint-model-specific signals feed into the "lower" score before
    thresholding: anatomically-implausible leg-joint clusters are discredited
    (see `_discount_collapsed_leg_keypoints`), and the box's own aspect ratio is
    blended in as an independent prior on whether legs are even in frame (see
    `_aspect_full_score`). Keypoint evidence remains the dominant signal in both.
    """
    points = list(keypoints)
    names = np.asarray([p.name for p in points], dtype=object)
    xy = np.asarray([(p.x, p.y) for p in points], dtype=np.float32).reshape(-1, 2)
    visibility = np.asarray([p.visibility for p in points], dtype=np.float32)
    # Leverage what only a pose model (not a plain box detector) can tell us:
    # discredit anatomically-impossible leg-joint clusters before they ever
    # reach the scoring math below.
    visibility = _discount_collapsed_leg_keypoints(names, xy, visibility, person_box, config)
    scores, evidence = {}, {}
    for label, group in (("face", FACE), ("upper", UPPER), ("lower", LOWER)):
        scores[label], evidence[label] = _group_score(names, xy, visibility, group, person_box, image_shape, config)
    confidence_scale = float(np.clip(pose_confidence, 0.0, 1.0))
    scores = {key: round(value * confidence_scale, 3) for key, value in scores.items()}
    # Box shape is a second, keypoint-independent vote on whether legs are even
    # in frame: it can rescue a moderate-confidence real leg detection, or pull
    # down a spurious one in an obviously torso-only crop. Keypoints still decide
    # (evidence["lower"]["visible"] >= 2 below is untouched by this blend).
    aspect_score = round(_aspect_full_score(person_box, config), 3)
    scores["lower"] = round((1 - config.lower_aspect_weight) * scores["lower"] + config.lower_aspect_weight * aspect_score, 3)
    scores["aspect_full"] = aspect_score
    small = min(person_box[2] - person_box[0], person_box[3] - person_box[1]) < config.min_person_size
    face = scores["face"] >= config.face_score and evidence["face"]["visible"] >= 2
    upper = scores["upper"] >= config.upper_score and evidence["upper"]["visible"] >= 2
    lower = scores["lower"] >= config.lower_score and evidence["lower"]["visible"] >= 2
    if small:
        upper = lower = False
    full_geometry_ok = aspect_score >= config.full_min_aspect_score
    if upper and lower and full_geometry_ok:
        body_visibility = "full"
    elif upper and lower and not full_geometry_ok:
        body_visibility = "upper_only"
    elif upper and not lower:
        body_visibility = "upper_only"
    elif lower and not upper:
        body_visibility = "lower_only"
    elif face:
        body_visibility = "face_only"
    else:
        body_visibility = "body_only"
    high_confidence_full = (upper and lower and full_geometry_ok and
                            scores["upper"] >= config.full_score and scores["lower"] >= config.full_score)
    return {"body_visibility": body_visibility, "scores": scores, "evidence": evidence,
            "ambiguous": bool(body_visibility == "full" and not high_confidence_full)}


def image_metadata(image: np.ndarray, min_dimension: int = 320, blur_threshold: float = 100.0) -> dict:
    """Return frame metadata; Laplacian variance <100 is a common blur baseline."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 100 is the commonly published variance-of-Laplacian starting threshold;
    # calibrate per camera/scene (PyImageSearch, 2015:
    # https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/).
    return {"width": width, "height": height, "aspect_ratio": width / height if height else 0.0,
            "megapixels": width * height / 1_000_000, "blur_score": blur_score,
            "usable": blur_score >= blur_threshold, "brightness_mean": float(gray.mean()),
            "is_low_resolution": min(width, height) < min_dimension}


def _visible(keypoints: Iterable[PoseKeypoint], group: set[str], threshold: float) -> int:
    return sum(point.name in group and point.visibility >= threshold for point in keypoints)


def body_visibility_category(keypoints: Iterable[PoseKeypoint], visibility: float, face_min: int, upper_min: int, lower_min: int) -> str:
    """Backward-compatible compact label for callers without frame geometry."""
    points = list(keypoints)
    face = _visible(points, FACE, visibility) >= face_min
    upper = _visible(points, UPPER, visibility) >= upper_min
    lower = _visible(points, LOWER, visibility) >= lower_min
    if upper and lower: return "full"
    if upper: return "upper_only"
    if lower: return "lower_only"
    if face: return "face_only"
    return "body_only"


def head_orientation(keypoints: Iterable[PoseKeypoint], visibility: float) -> str:
    """Coarse 2-D yaw heuristic, not gaze: true pose fits a 3-D face model (PnP)."""
    # Sparse landmarks cannot yield robust pose: MediaPipe uses 468 3-D points and
    # a face transform (https://developers.google.com/mediapipe/solutions/vision/face_landmarker).
    # This adapts the lightweight 2-D alternative: compare nose displacement from
    # the eye midpoint, and require visible far-side landmarks before a non-profile label.
    points = {p.name: p for p in keypoints if p.visibility >= visibility}
    eyes = [points.get("left_eye"), points.get("right_eye")]
    sides = [points.get("left_face"), points.get("right_face")]
    nose = points.get("nose")
    if not nose and not any(eyes):
        return "away" if all(sides) else "undetermined"
    if not nose or not all(eyes):
        return "profile" if nose and (any(eyes) or any(sides)) else "undetermined"
    left, right = eyes
    midpoint = (left.x + right.x) / 2
    span = abs(right.x - left.x)
    if span < 1e-6: return "undetermined"
    offset = (nose.x - midpoint) / span
    if abs(offset) <= 0.18: return "facing_camera"
    if abs(offset) >= 0.55 or not all(sides): return "profile"
    return "turned_left" if offset < 0 else "turned_right"


CAMERA_ORIENTATION_CLASSES = ("facing_camera", "facing_away", "insufficient_evidence")

_FACING_AWAY_ORIENTATIONS = {"turned_left", "turned_right", "profile", "away"}


def summarize_camera_orientation(orientation: str) -> str:
    """Collapse `head_orientation`'s six-way yaw estimate into the three-state
    signal the unified per-person output actually needs: is this person facing
    the camera, facing away/to the side, or is there not enough face evidence
    to say either way. The detailed value is still available separately for
    anyone who wants it -- this doesn't replace `head_orientation`, it summarizes it.
    """
    if orientation == "facing_camera":
        return "facing_camera"
    if orientation in _FACING_AWAY_ORIENTATIONS:
        return "facing_away"
    return "insufficient_evidence"
