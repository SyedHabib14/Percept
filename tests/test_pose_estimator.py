import numpy as np

from pdc.pose_estimator import decode_output


def test_decode_inspected_57_by_8400_contract():
    raw = np.zeros((1, 57, 8400), dtype=np.float16)
    raw[0, :6, 7] = [320, 320, 200, 400, .9, .1]
    for point in range(17):
        raw[0, 6 + point * 3:9 + point * 3, 7] = [320 + point, 200 + point, .8]
    poses = decode_output(raw, (640, 640), .25, .45, 1.0, 0.0, 0.0)
    assert len(poses) == 1
    pose = poses[0]
    assert pose.class_name == "person" and pose.box == (220, 120, 420, 520)
    assert len(pose.keypoints) == 17
    assert pose.keypoints[0].name == "nose" and pose.keypoints[-1].name == "right_ankle"
    assert pose.keypoints[0].visibility == np.float16(.8)


def test_decode_deletterboxes_coordinates():
    raw = np.zeros((1, 57, 8400), dtype=np.float16)
    raw[0, :6, 0] = [320, 320, 200, 200, .9, 0]
    decoded = decode_output(raw, (320, 320), .25, .45, 2.0, 0.0, 0.0)
    assert decoded[0].box == (110, 110, 210, 210)
