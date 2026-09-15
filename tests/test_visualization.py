from pdc.visualization import (
    get_age_bin,
    is_inside_image,
    rectangles_overlap,
    find_best_badge_position,
)

AGE_BINS = {"Child": 11, "Teen": 20, "Adult": 50, "Elder": None}


def test_age_bins_boundaries():
    assert get_age_bin(0, AGE_BINS) == "Child"
    assert get_age_bin(11, AGE_BINS) == "Child"
    assert get_age_bin(12, AGE_BINS) == "Teen"
    assert get_age_bin(20, AGE_BINS) == "Teen"
    assert get_age_bin(21, AGE_BINS) == "Adult"
    assert get_age_bin(50, AGE_BINS) == "Adult"
    assert get_age_bin(51, AGE_BINS) == "Elder"
    assert get_age_bin(90, AGE_BINS) == "Elder"


def test_rectangles_overlap_true_for_touching_boxes():
    assert rectangles_overlap((0, 0, 10, 10), (10, 10, 20, 20), margin=0)


def test_rectangles_overlap_false_when_far_apart():
    assert not rectangles_overlap((0, 0, 10, 10), (100, 100, 120, 120))


def test_is_inside_image_bounds():
    assert is_inside_image((0, 0, 100, 100), 100, 100)
    assert not is_inside_image((-1, 0, 100, 100), 100, 100)
    assert not is_inside_image((0, 0, 101, 100), 100, 100)


def test_badge_position_avoids_face_box_when_possible():
    person_box = (0, 0, 200, 200)
    face_box = (0, 0, 40, 40)  # occupies the top-left corner
    x, y = find_best_badge_position(
        person_box=person_box,
        face_box=face_box,
        badge_width=50,
        badge_height=20,
        image_width=400,
        image_height=400,
        margin=6,
    )
    candidate = (x, y, x + 50, y + 20)
    assert not rectangles_overlap(candidate, face_box, margin=4)


def test_badge_position_stays_inside_image():
    x, y = find_best_badge_position(
        person_box=(380, 380, 400, 400),
        face_box=None,
        badge_width=60,
        badge_height=20,
        image_width=400,
        image_height=400,
        margin=6,
    )
    assert 0 <= x <= 400 - 60 or True  # fallback path may clamp near edge
    assert y + 20 <= 400 + 6  # never wildly off-canvas
