import numpy as np

from pdc.metadata import BodyVisibilityConfig, body_visibility_category, classify_body_visibility, head_orientation, image_metadata, summarize_camera_orientation
from pdc.pose_estimator import KEYPOINT_NAMES, PoseKeypoint


def points(visible=()):
    return [PoseKeypoint(name, float(index * 10), 10.0, 1.0 if name in visible else 0.0)
            for index, name in enumerate(KEYPOINT_NAMES)]


def test_body_visibility_categories():
    face = {"nose", "left_eye"}
    upper = {"left_shoulder", "right_shoulder", "left_elbow"}
    lower = {"left_knee", "right_knee"}
    arguments = (0.5, 2, 3, 2)
    assert body_visibility_category(points(face), *arguments) == "face_only"
    assert body_visibility_category(points(face | upper), *arguments) == "upper_only"
    assert body_visibility_category(points(face | lower), *arguments) == "lower_only"
    assert body_visibility_category(points(face | upper | lower), *arguments) == "full"
    assert body_visibility_category(points(upper), *arguments) == "upper_only"
    assert body_visibility_category(points({"nose"}), *arguments) == "body_only"


def make_pose(visible, box=(0, 0, 400, 800), visibility=1.0):
    values = {name: (100 + i * 10, 100 + i * 20) for i, name in enumerate(KEYPOINT_NAMES)}
    values.update({"left_hip": (120, 300), "right_hip": (180, 300),
                   "left_knee": (120, 500), "right_knee": (180, 500),
                   "left_ankle": (120, 700), "right_ankle": (180, 700)})
    return [PoseKeypoint(name, *values[name], visibility if name in visible else 0.0) for name in KEYPOINT_NAMES]


def test_full_body_moderate_confidence_is_not_collapsed_to_body_only():
    """Regression test: upper AND lower both individually detected with real,
    believable confidence (0.6 -- clears the 0.56 per-region bar but not the
    stricter 0.68 full_score bar) must still be classified 'full', just flagged
    'ambiguous' for lower confidence. It must never be relabeled 'body_only',
    which is a different class (torso only, no limbs) and throws the detected
    limb evidence away entirely. Box is a plausible full-body shape (aspect 3.0)
    so the aspect-ratio prior doesn't itself dominate the outcome here."""
    upper = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip"}
    lower = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
    box, shape = (0, 0, 300, 900), (900, 700)

    moderate = classify_body_visibility(make_pose(upper | lower, visibility=0.6), box, shape)
    assert moderate["body_visibility"] == "full"
    assert moderate["ambiguous"] is True

    high = classify_body_visibility(make_pose(upper | lower, visibility=1.0), box, shape)
    assert high["body_visibility"] == "full"
    assert high["ambiguous"] is False


def _leg_pose(collapsed: bool):
    """Same person, same box; the only difference is whether the leg keypoints
    are real (well-separated down the leg) or a pose-model hallucination
    pattern (hip/knee/ankle all bunched at nearly the same point)."""
    box = (0, 0, 200, 700)
    points = []
    for i, name in enumerate(("left_shoulder", "right_shoulder", "left_elbow",
                               "right_elbow", "left_wrist", "right_wrist")):
        points.append(PoseKeypoint(name, 50 + i * 10, 200 + i * 5, 1.0))
    if collapsed:
        points += [
            PoseKeypoint("left_hip", 50, 300, 0.9), PoseKeypoint("left_knee", 52, 302, 0.9),
            PoseKeypoint("left_ankle", 54, 304, 0.9),
            PoseKeypoint("right_hip", 150, 300, 0.9), PoseKeypoint("right_knee", 152, 302, 0.9),
            PoseKeypoint("right_ankle", 154, 304, 0.9),
        ]
    else:
        points += [
            PoseKeypoint("left_hip", 50, 300, 0.9), PoseKeypoint("left_knee", 50, 500, 0.9),
            PoseKeypoint("left_ankle", 50, 690, 0.9),
            PoseKeypoint("right_hip", 150, 300, 0.9), PoseKeypoint("right_knee", 150, 500, 0.9),
            PoseKeypoint("right_ankle", 150, 690, 0.9),
        ]
    for name in ("nose", "left_eye", "right_eye", "left_face", "right_face"):
        points.append(PoseKeypoint(name, 100, 120, 0.0))
    return points, box


def test_collapsed_leg_keypoints_are_discredited_not_trusted():
    """Regression test for the anatomical-plausibility check: a hip/knee/ankle
    cluster that's physically impossible for a real human body (all within a
    few px of each other) must not count as lower-body evidence, even though
    each individual point reports high confidence. A real, well-separated leg
    with the same per-point confidence must still be counted normally."""
    box, shape = (0, 0, 200, 700), (800, 600)

    collapsed_points, _ = _leg_pose(collapsed=True)
    collapsed = classify_body_visibility(collapsed_points, box, shape)
    assert collapsed["evidence"]["lower"]["visible"] == 0
    assert collapsed["body_visibility"] != "full"

    real_points, _ = _leg_pose(collapsed=False)
    real = classify_body_visibility(real_points, box, shape)
    assert real["evidence"]["lower"]["visible"] == 4
    assert real["body_visibility"] == "full"


def _pose_in_box(visibilities: dict, box):
    """Like `_pose_in_box` above but with a per-keypoint-name visibility map,
    so upper- and lower-body confidence can be set independently."""
    x1, y1, x2, y2 = box
    cx, height = (x1 + x2) / 2, max(1.0, y2 - y1)
    return [
        PoseKeypoint(name, cx, y1 + (index + 1) / (len(KEYPOINT_NAMES) + 1) * height, visibilities.get(name, 0.0))
        for index, name in enumerate(KEYPOINT_NAMES)
    ]


def test_aspect_ratio_rescues_moderate_confidence_legs_in_a_tall_crop():
    """A weak-but-real leg-keypoint score (below the lower_score bar on its own)
    in a box shaped like a full standing body should be pulled over the bar by
    the aspect-ratio prior; the identical keypoint evidence in a short, clearly
    torso-only-shaped box should not. Upper-body confidence is kept high and
    fixed in both cases so only the "lower" signal is under test."""
    upper = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip"}
    lower = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
    visibilities = {**{name: 0.9 for name in upper}, **{name: 0.5 for name in lower}}
    shape = (900, 700)

    tall_box = (0, 0, 200, 700)      # aspect 3.5 -> strongly full-body-shaped
    short_box = (0, 0, 400, 800)     # aspect 2.0 -> at the "no boost" edge

    tall = classify_body_visibility(_pose_in_box(visibilities, tall_box), tall_box, shape)
    short = classify_body_visibility(_pose_in_box(visibilities, short_box), short_box, shape)

    assert tall["scores"]["lower"] > short["scores"]["lower"]
    assert tall["body_visibility"] == "full"
    assert short["body_visibility"] != "full"


def test_classifier_all_classes_and_low_confidence_failure():
    face = {"nose", "left_eye", "right_eye"}
    upper = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}
    lower = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
    box, shape = (0, 0, 300, 900), (900, 700)
    assert classify_body_visibility(make_pose(face), box, shape)["body_visibility"] == "face_only"
    assert classify_body_visibility(make_pose(upper), box, shape)["body_visibility"] == "upper_only"
    assert classify_body_visibility(make_pose(lower), box, shape)["body_visibility"] == "lower_only"
    assert classify_body_visibility(make_pose(upper | lower), box, shape)["body_visibility"] == "full"
    assert classify_body_visibility(make_pose(upper | lower, visibility=.2), box, shape)["body_visibility"] == "body_only"


def test_torso_shaped_box_cannot_become_full_from_hallucinated_legs():
    upper = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}
    lower = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
    result = classify_body_visibility(make_pose(upper | lower), (0, 0, 800, 534), (534, 800))
    assert result["body_visibility"] == "upper_only"
    assert result["scores"]["aspect_full"] == 0.0


def face_points(values):
    return [PoseKeypoint(name, x, 0.0, 1.0) for name, x in values.items()]


def test_head_orientation_branches():
    both_sides = {"left_face": 0, "right_face": 10}
    assert head_orientation(face_points({"left_eye": 0, "right_eye": 10, "nose": 5, **both_sides}), .5) == "facing_camera"
    assert head_orientation(face_points({"left_eye": 0, "right_eye": 10, "nose": 2, **both_sides}), .5) == "turned_left"
    assert head_orientation(face_points({"left_eye": 0, "right_eye": 10, "nose": 8, **both_sides}), .5) == "turned_right"
    assert head_orientation(face_points({"nose": 5, "left_eye": 0}), .5) == "profile"
    assert head_orientation(face_points(both_sides), .5) == "away"
    assert head_orientation([], .5) == "undetermined"


CAMERA_ORIENTATION_CLASSES = ("facing_camera", "facing_away", "insufficient_evidence")


def test_camera_orientation_summarizes_to_three_states():
    assert summarize_camera_orientation("facing_camera") == "facing_camera"
    for detailed in ("turned_left", "turned_right", "profile", "away"):
        assert summarize_camera_orientation(detailed) == "facing_away"
    assert summarize_camera_orientation("undetermined") == "insufficient_evidence"
    for value in ("facing_camera", "turned_left", "turned_right", "profile", "away", "undetermined"):
        assert summarize_camera_orientation(value) in CAMERA_ORIENTATION_CLASSES


def test_image_metadata_black_white_and_normal():
    black = image_metadata(np.zeros((20, 40, 3), dtype=np.uint8))
    white = image_metadata(np.full((20, 40, 3), 255, dtype=np.uint8))
    checker = np.indices((20, 40)).sum(axis=0) % 2 * 255
    normal = image_metadata(np.repeat(checker[..., None], 3, axis=2).astype(np.uint8), min_dimension=10)
    assert black["brightness_mean"] == 0 and black["blur_score"] == 0
    assert white["brightness_mean"] == 255 and white["blur_score"] == 0
    assert normal["blur_score"] > 0 and not normal["is_low_resolution"]
