"""Shared dataclasses used by both `pipeline.py` and `visualization.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pdc.analyzer import AttributeResult
from pdc.detector import Detection
from pdc.pose_estimator import PoseResult


@dataclass
class PersonAnalysis:
    person_index: int
    person: Detection
    face: Optional[Detection]
    analysis: Optional[AttributeResult]
    pose: Optional[PoseResult] = None
    metadata: Optional[dict] = None
    track_id: Optional[int] = None
    raw_age: Optional[float] = None
    smoothed_age: Optional[float] = None
    person_id: Optional[str] = None

    def as_dict(self, include_metadata: bool = False) -> dict:
        result = {
            "person_id": self.person_id,
            "person_index": self.person_index,
            "person": self.person.as_dict(),
            "face": self.face.as_dict() if self.face is not None else None,
            "analysis": self.analysis.as_dict() if self.analysis is not None else None,
            "visibility_class": (self.metadata or {}).get("visibility_class", "BO"),
            "body_visibility": (self.metadata or {}).get("body_visibility", "body_only"),
            "visibility_scores": (self.metadata or {}).get("scores", {}),
            "visibility_evidence": (self.metadata or {}).get("evidence", {}),
            "camera_orientation": (self.metadata or {}).get("camera_orientation", "insufficient_evidence"),
        }
        if include_metadata:
                result.update({"pose": _pose_as_dict(self.pose), "metadata": self.metadata,
                           "track_id": self.track_id, "raw_age": self.raw_age,
                           "smoothed_age": self.smoothed_age})
        return result


def _pose_as_dict(pose: Optional[PoseResult]) -> Optional[dict]:
    if pose is None:
        return None
    return {"box": pose.box, "confidence": pose.confidence, "class_name": pose.class_name,
            "keypoints": [{"name": p.name, "x": p.x, "y": p.y, "visibility": p.visibility} for p in pose.keypoints]}