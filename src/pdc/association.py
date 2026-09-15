"""
Associating detected faces with detected people.

The pipeline's primary strategy — matching each person to the largest
face whose *center point* falls inside that person's box — is ported
verbatim from the notebook's analysis loop, since that is the logic
that actually produced the notebook's results.

`calculate_iou` / `associate_faces_with_people` (best-IoU matching) is
also provided as an alternative strategy: it was defined in the
notebook but not on the code path that ran, and is kept here for
callers who want IoU-based association instead (e.g. dense crowds
where "largest contained face" is too permissive).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pdc.detector import Detection

Box = Tuple[int, int, int, int]


def calculate_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def associate_faces_with_people(persons: List[Detection], faces: List[Detection]) -> List[dict]:
    """Alternative strategy: best-IoU match per person. See module docstring."""
    associations = []
    for person_index, person in enumerate(persons):
        best_face, best_iou = None, 0.0
        for face in faces:
            iou = calculate_iou(person.box, face.box)
            if iou > best_iou:
                best_iou, best_face = iou, face
        associations.append(
            {
                "person_index": person_index,
                "person": person,
                "face": best_face,
                "association_score": best_iou,
            }
        )
    return associations


def match_face_to_person(person_box: Box, faces: List[Detection]) -> Optional[Detection]:
    """
    Primary strategy used by the pipeline: among faces whose center
    point lies inside `person_box`, pick the one with the largest area.
    """
    px1, py1, px2, py2 = person_box
    best_face: Optional[Detection] = None
    best_area = -1.0

    for face in faces:
        fx1, fy1, fx2, fy2 = face.box
        center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2

        if not (px1 <= center_x <= px2 and py1 <= center_y <= py2):
            continue

        area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        if area > best_area:
            best_area, best_face = area, face

    return best_face
