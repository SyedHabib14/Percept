"""Pure UI helpers shared by the Streamlit console and its tests."""

from __future__ import annotations

from typing import Iterable, Optional


def person_id_for_index(index: int) -> str:
    """Return the stable, operator-facing identifier for a zero-based index."""
    return f"PERSON {index + 1:02d}"


def person_at_point(results: Iterable[object], x: int, y: int) -> Optional[str]:
    """Find the smallest detected person box containing a clicked image point.

    Choosing the smallest containing box makes nested/overlapping detections
    deterministic and keeps the image-to-card link predictable.
    """
    matches: list[tuple[int, object]] = []
    for result in results:
        x1, y1, x2, y2 = result.person.box
        if x1 <= x <= x2 and y1 <= y <= y2:
            matches.append(((x2 - x1) * (y2 - y1), result))
    if not matches:
        return None
    selected = min(matches, key=lambda item: item[0])[1]
    return selected.person_id or person_id_for_index(selected.person_index)


def selection_is_valid(results: Iterable[object], person_id: Optional[str]) -> bool:
    """Whether a selection belongs to the inference result currently shown."""
    return person_id is not None and any(
        (item.person_id or person_id_for_index(item.person_index)) == person_id
        for item in results
    )
