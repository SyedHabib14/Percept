"""
Regression tests for the live-mode `body_visibility` temporal debounce in
`PDCPipeline._analyze_people`.

Root cause being guarded against: the debounce history used to be keyed on
`tuple(person.box)` -- the raw, per-frame pixel box. Real pose-model output
jitters by a few pixels every frame even for a stationary subject, so that key
almost never repeats between frames. The "majority of the last 3 frames" vote
therefore never accumulated more than one entry and never overrode a noisy
single-frame misclassification, even though the code looked like it should.

The fix keys the history on the IoU-tracked `track_id` (the same tracker
`AgeSmoother` already uses for age smoothing) instead of the raw box.

These tests build a `PDCPipeline` without running its real `__init__` (which
would try to load ONNX/YOLO weights from disk), and drive `_analyze_people`
directly with lightweight fakes, calling it repeatedly to simulate consecutive
video frames -- exactly how `process_image` calls it once per frame.
"""

from __future__ import annotations

import numpy as np

from pdc.age_smoother import AgeSmoother
from pdc.analyzer import AttributeResult
from pdc.config import PDCConfig
from pdc.detector import Detection
from pdc.pipeline import PDCPipeline
from pdc.pose_estimator import KEYPOINT_NAMES, PoseKeypoint, PoseResult

UPPER_NAMES = {
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
}
LOWER_NAMES = {"left_knee", "right_knee", "left_ankle", "right_ankle"}


class _FakeAnalyzer:
    """Stands in for AttributeAnalyzer so no ONNX model needs to be loaded."""

    def analyze(self, face_crop, person_crop):
        return AttributeResult(
            estimated_age=30.0, gender_probability=0.5, category_index=0, face_available=False
        )


def _make_bare_pipeline() -> PDCPipeline:
    pipeline = PDCPipeline.__new__(PDCPipeline)  # skip __init__: no weights on disk in this test
    pipeline.config = PDCConfig()
    pipeline.analyzer = _FakeAnalyzer()
    pipeline.age_smoother = AgeSmoother(window=5, iou_threshold=0.2, max_age_seconds=30.0)
    pipeline._visibility_history = {}
    return pipeline


def _keypoints_for(box, upper_visible: bool, lower_visible: bool) -> list[PoseKeypoint]:
    """Spread all 17 points vertically down the box (KEYPOINT_NAMES is already
    roughly head-to-toe ordered), matching real anatomy well enough that the
    leg-collapse plausibility check in metadata.py doesn't misfire -- a
    fixture that bunched every point at one coordinate would itself look like
    the exact hallucination pattern that check is designed to catch."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    height = max(1.0, y2 - y1)
    keypoints = []
    for index, name in enumerate(KEYPOINT_NAMES):
        y = y1 + (index + 1) / (len(KEYPOINT_NAMES) + 1) * height
        visible = (name in UPPER_NAMES and upper_visible) or (name in LOWER_NAMES and lower_visible)
        keypoints.append(PoseKeypoint(name, cx, y, 1.0 if visible else 0.0))
    return keypoints


def _frame(box, upper_visible=True, lower_visible=True):
    persons = [Detection(box=box, confidence=0.9, class_name="person")]
    poses = [PoseResult(keypoints=_keypoints_for(box, upper_visible, lower_visible),
                         box=box, confidence=0.9, class_name="person")]
    return persons, poses


def _run_frames(pipeline, image, frame_specs):
    labels = []
    for box, upper_visible, lower_visible in frame_specs:
        persons, poses = _frame(box, upper_visible, lower_visible)
        results = pipeline._analyze_people(image, persons, [], poses, live=True)
        labels.append(results[0].metadata["body_visibility"])
    return labels


def test_visibility_debounce_smooths_a_single_noisy_frame_across_jittering_boxes():
    pipeline = _make_bare_pipeline()
    image = np.zeros((800, 600, 3), dtype=np.uint8)

    # Same stationary person; the box jitters by a few px every frame, exactly
    # like real detector output. Frame index 2 is a single noisy frame where
    # the lower-body keypoints briefly drop out (e.g. motion blur) -- every
    # other frame is a clean, unambiguous full-body detection.
    frames = [
        ((100, 200, 300, 700), True, True),
        ((101, 199, 301, 701), True, True),
        ((99, 201, 299, 699), True, False),   # noisy frame: raw label is "upper_only"
        ((102, 200, 300, 700), True, True),
        ((100, 198, 302, 702), True, True),
    ]
    labels = _run_frames(pipeline, image, frames)

    assert labels[0] == "full"
    assert labels[1] == "full"
    # Raw per-frame label here is "upper_only", but the last-3-frames majority
    # vote (full, full, upper_only) must resolve it to "full".
    assert labels[2] == "full"
    assert labels[3] == "full"
    assert labels[4] == "full"

    # Sanity: this whole sequence was a single tracked person, one track only.
    assert len(pipeline.age_smoother.active_track_ids()) == 1


def test_person_id_and_camera_orientation_are_wired_through_as_dict():
    """Architecture check: each person in the unified per-frame output must
    carry a stable, human-readable id ('person_001', ...) and a summarized
    camera_orientation, alongside the existing body_visibility fields."""
    pipeline = _make_bare_pipeline()
    image = np.zeros((800, 600, 3), dtype=np.uint8)

    persons, poses = _frame((100, 200, 300, 700), upper_visible=True, lower_visible=True)
    results = pipeline._analyze_people(image, persons, [], poses, live=False)

    payload = results[0].as_dict()
    assert payload["person_id"] == "person_001"
    assert payload["camera_orientation"] in ("facing_camera", "facing_away", "insufficient_evidence")

    # A second person in the same frame gets the next id.
    persons2, poses2 = _frame((400, 200, 550, 700), upper_visible=True, lower_visible=False)
    combined_persons = persons + persons2
    combined_poses = poses + poses2
    results2 = pipeline._analyze_people(image, combined_persons, [], combined_poses, live=False)
    assert [r.as_dict()["person_id"] for r in results2] == ["person_001", "person_002"]


def test_visibility_history_does_not_leak_across_different_people():
    pipeline = _make_bare_pipeline()
    image = np.zeros((800, 600, 3), dtype=np.uint8)

    persons_a, poses_a = _frame((100, 200, 300, 700), upper_visible=True, lower_visible=True)
    pipeline._analyze_people(image, persons_a, [], poses_a, live=True)

    # A second, unrelated person (far away, no IoU overlap) must get its own
    # track and its own history, not inherit the first person's votes.
    persons_b, poses_b = _frame((400, 200, 550, 700), upper_visible=True, lower_visible=False)
    result_b = pipeline._analyze_people(image, persons_b, [], poses_b, live=True)

    assert result_b[0].metadata["body_visibility"] == "upper_only"
    assert len(pipeline.age_smoother.active_track_ids()) == 2