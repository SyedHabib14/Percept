from pathlib import Path
from types import SimpleNamespace

from pdc.config import PDCConfig
from pdc.ui_state import person_at_point, person_id_for_index, selection_is_valid


def _person(index, box, person_id=None):
    return SimpleNamespace(
        person_index=index,
        person_id=person_id,
        person=SimpleNamespace(box=box),
    )


def test_operator_person_ids_are_stable_and_one_based():
    assert person_id_for_index(0) == "PERSON 01"
    assert person_id_for_index(2) == "PERSON 03"


def test_image_click_selects_person_containing_coordinate():
    people = [_person(0, (10, 10, 80, 90)), _person(1, (100, 20, 180, 120))]
    assert person_at_point(people, 120, 45) == "PERSON 02"
    assert person_at_point(people, 99, 45) is None


def test_overlapping_boxes_choose_smallest_and_result_selection_is_scoped():
    outer = _person(0, (0, 0, 100, 100), "PERSON 01")
    inner = _person(1, (20, 20, 50, 50), "PERSON 02")
    assert person_at_point([outer, inner], 30, 30) == "PERSON 02"
    assert selection_is_valid([outer, inner], "PERSON 02")
    assert not selection_is_valid([outer, inner], "PERSON 03")


def test_default_config_resolves_model_paths_from_repo_root():
    cfg = PDCConfig.load()

    assert Path(cfg.model.detection_weights).is_absolute()
    assert Path(cfg.model.analysis_weights).is_absolute()
    assert Path(cfg.model.pose_weights).is_absolute()
    assert Path(cfg.model.detection_weights).exists()
    assert Path(cfg.model.analysis_weights).exists()
    assert Path(cfg.model.pose_weights).exists()
