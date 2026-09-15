from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionSchema(BaseModel):
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    confidence: float
    class_name: str


class AttributeSchema(BaseModel):
    estimated_age: float
    gender_probability: float
    category_index: int
    gender_label: str
    age_band: str
    face_available: bool


class PersonResultSchema(BaseModel):
    person_id: str = Field(..., description="Stable per-frame display id, e.g. 'person_001'")
    person_index: int
    person: DetectionSchema
    face: Optional[DetectionSchema]
    analysis: Optional[AttributeSchema]
    visibility_class: str = Field(..., description="UB, LB, FB, FO, or BO")
    body_visibility: str = Field(..., description="face_only, upper_only, lower_only, full, or body_only")
    visibility_scores: dict
    visibility_evidence: dict
    camera_orientation: str = Field(..., description="facing_camera, facing_away, or insufficient_evidence")


class InferenceResponse(BaseModel):
    person_count: int
    face_count: int
    people: List[PersonResultSchema]
    timing_ms: dict
    annotated_image_base64: Optional[str] = Field(
        None, description="Base64 JPEG of the annotated image, included when include_image=true"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    analysis_providers: List[str]