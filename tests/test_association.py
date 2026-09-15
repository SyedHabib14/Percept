from pdc.association import calculate_iou, match_face_to_person
from pdc.detector import Detection


def test_iou_identical_boxes_is_one():
    box = (0, 0, 100, 100)
    assert calculate_iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_partial_overlap():
    iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
    # intersection = 5x5=25, union = 100+100-25=175
    assert abs(iou - 25 / 175) < 1e-9


def test_match_face_to_person_picks_largest_contained_face():
    person_box = (0, 0, 200, 200)
    small_face = Detection(box=(10, 10, 30, 30), confidence=0.9, class_name="face")
    large_face = Detection(box=(50, 50, 120, 120), confidence=0.8, class_name="face")
    outside_face = Detection(box=(500, 500, 550, 550), confidence=0.95, class_name="face")

    best = match_face_to_person(person_box, [small_face, large_face, outside_face])
    assert best is large_face


def test_match_face_to_person_returns_none_when_no_face_center_inside():
    person_box = (0, 0, 50, 50)
    face = Detection(box=(100, 100, 150, 150), confidence=0.9, class_name="face")
    assert match_face_to_person(person_box, [face]) is None


def test_match_face_to_person_with_no_faces():
    assert match_face_to_person((0, 0, 50, 50), []) is None
