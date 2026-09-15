"""
Centralized, typed configuration for the PDC pipeline.

Every tunable in the original research notebook (confidence thresholds,
input sizes, drawing constants) lives here so behavior can be changed
without touching pipeline code, and so the exact same config drives
image, batch, webcam, and API inference paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


@dataclass
class ModelConfig:
    """Model weights, thresholds, and runtime device selection."""

    detection_weights: str = "models/detection_engine.pt"
    analysis_weights: str = "models/analysis_engine.onnx"
    pose_weights: str = "models/dpose_engine.onnx"

    # Optional remote sources used by `pdc.model_hub` when a local file
    # is missing, so the same config works on a fresh machine.
    detection_weights_url: Optional[str] = None
    analysis_weights_url: Optional[str] = None
    pose_weights_url: Optional[str] = None
    hf_repo_id: Optional[str] = None  # e.g. "your-org/pdc-weights"

    detection_confidence: float = 0.25
    detection_iou: float = 0.45
    analysis_input_size: int = 384
    pose_keypoint_visibility: float = 0.5
    body_face_min_visible: int = 2
    body_upper_min_visible: int = 3
    body_lower_min_visible: int = 2
    min_image_dimension: int = 320

    # "cpu", "cuda", or "auto" (use CUDA if onnxruntime-gpu + a GPU are present)
    device: str = "auto"

    person_class_name: str = "person"
    face_class_name: str = "face"


@dataclass
class VisualizationConfig:
    """Drawing constants — identical defaults to the source notebook."""

    person_line_width: int = 2
    face_line_width: int = 2
    badge_alpha: int = 225
    badge_margin: int = 6
    gender_labels: dict = field(default_factory=lambda: {0: "♂️", 1: "♀️"})
    age_bins: dict = field(
        default_factory=lambda: {"Child": 11, "Teen": 20, "Adult": 50, "Elder": None}
    )
    age_display_mode: str = "binned"  # "exact" or "binned"
    age_smoothing: str = "none"  # "none" or "rolling_average"; live only
    age_smoothing_window: int = 5


@dataclass
class WebcamConfig:
    """Live-inference tuning: source, skip-frame rate, overlay behavior."""

    source: str = "0"  # camera index, video file path, or RTSP URL
    process_every_n_frames: int = 1
    mirror: bool = True
    show_fps: bool = True
    window_title: str = "PDC — Live Analysis"
    record_output: Optional[str] = None  # e.g. "output.mp4"


@dataclass
class PDCConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    webcam: WebcamConfig = field(default_factory=WebcamConfig)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "PDCConfig":
        """
        Load config from YAML, falling back to `config/default.yaml`
        and then to hard-coded defaults. Environment variables prefixed
        with `PDC_` override individual model fields, e.g.
        `PDC_DEVICE=cuda`, `PDC_DETECTION_WEIGHTS=/models/det.pt`.
        """
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        data = {}
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

        cfg = cls(
            model=ModelConfig(**data.get("model", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
            webcam=WebcamConfig(**data.get("webcam", {})),
        )
        cfg._apply_env_overrides(path.parent)
        cfg._resolve_model_paths(path.parent)
        return cfg

    def _apply_env_overrides(self, base_dir: Path) -> None:
        env_map = {
            "PDC_DETECTION_WEIGHTS": ("model", "detection_weights"),
            "PDC_ANALYSIS_WEIGHTS": ("model", "analysis_weights"),
            "PDC_POSE_WEIGHTS": ("model", "pose_weights"),
            "PDC_DEVICE": ("model", "device"),
            "PDC_DETECTION_CONFIDENCE": ("model", "detection_confidence"),
            "PDC_DETECTION_IOU": ("model", "detection_iou"),
            "PDC_HF_REPO_ID": ("model", "hf_repo_id"),
            # MODEL_REPO is the concise Hugging Face Space Secret name used by
            # the Streamlit deployment guide. Keep PDC_HF_REPO_ID for existing
            # CLI/API users.
            "MODEL_REPO": ("model", "hf_repo_id"),
        }
        for env_key, (section, field_name) in env_map.items():
            if env_key not in os.environ:
                continue
            value = os.environ[env_key]
            section_obj = getattr(self, section)
            current = getattr(section_obj, field_name)
            if isinstance(current, float):
                value = float(value)
            if field_name.endswith("weights"):
                value = self.resolve_path(value, base_dir)
            setattr(section_obj, field_name, value)

    def _resolve_model_paths(self, base_dir: Path) -> None:
        self.model.detection_weights = self.resolve_path(self.model.detection_weights, base_dir)
        self.model.analysis_weights = self.resolve_path(self.model.analysis_weights, base_dir)
        self.model.pose_weights = self.resolve_path(self.model.pose_weights, base_dir)

    def resolve_path(self, relative: str | Path, base_dir: Path | None = None) -> Path:
        p = Path(relative)
        if p.is_absolute():
            return p.resolve(strict=False)
        base = base_dir or REPO_ROOT
        return (base / p).resolve(strict=False)

    def as_dict(self) -> dict:
        return asdict(self)
