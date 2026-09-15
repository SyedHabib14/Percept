"""
Stage 2 — Attribute Analysis.

Wraps the ONNX Runtime session that estimates age + gender from a
face crop (when available) and a body crop, exactly matching the
notebook's two-input `faces_input` / `body_input` contract -- this is
the MiVOLO v2 dual-input architecture (face + body fusion for age/gender),
not a coincidence of the notebook's design.

Note on device selection: the original notebook's provider-selection
branch was a no-op (both branches resolved to CPUExecutionProvider
regardless of `device_preference`). This wrapper implements the
intended behavior — CUDA is used when requested and available, with a
safe fallback to CPU.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def _load_cuda_runtime_dlls() -> None:
    """Make pip-installed NVIDIA DLLs visible to ONNX Runtime on Windows."""
    if sys.platform != "win32":
        return

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return

    for dll_dir in nvidia_root.glob("*/bin"):
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, OSError):
            pass


_load_cuda_runtime_dlls()

import onnxruntime as ort

from pdc.config import ModelConfig
from pdc.model_hub import resolve_weight

logger = logging.getLogger("pdc.analyzer")


@dataclass
class AttributeResult:
    estimated_age: float
    gender_probability: float
    category_index: int
    face_available: bool

    def as_dict(self) -> dict:
        return {
            "estimated_age": self.estimated_age,
            "gender_probability": self.gender_probability,
            "category_index": self.category_index,
            "face_available": self.face_available,
        }


def _resolve_providers(device: str) -> List[str]:
    available = ort.get_available_providers()

    if device == "cpu":
        return ["CPUExecutionProvider"]

    if device in ("cuda", "gpu", "auto") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if device in ("cuda", "gpu"):
        logger.warning(
            "device=%r requested but CUDAExecutionProvider is unavailable; "
            "falling back to CPU.",
            device,
        )

    return ["CPUExecutionProvider"]


def preprocess_crop(crop: np.ndarray, size: int) -> np.ndarray:
    if crop is None:
        raise ValueError("Crop cannot be None.")

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = crop.astype(np.float32) / 255.0
    crop = np.transpose(crop, (2, 0, 1))
    crop = np.expand_dims(crop, axis=0)
    return np.ascontiguousarray(crop, dtype=np.float32)


def create_empty_face_input(size: int) -> np.ndarray:
    return np.zeros((1, 3, size, size), dtype=np.float32)


class AttributeAnalyzer:
    """Loads the ONNX analysis model once and exposes `.analyze()`."""

    def __init__(self, config: ModelConfig):
        self._config = config

        weights_path = resolve_weight(
            config.analysis_weights,
            url=config.analysis_weights_url,
            hf_repo_id=config.hf_repo_id,
        )
        providers = _resolve_providers(config.device)
        logger.info("Loading analysis model from %s (providers=%s)", weights_path, providers)
        self._session = ort.InferenceSession(str(weights_path), providers=providers)

    @property
    def active_providers(self) -> List[str]:
        return self._session.get_providers()

    def analyze(self, face_crop: Optional[np.ndarray], person_crop: np.ndarray) -> AttributeResult:
        size = self._config.analysis_input_size
        person_tensor = preprocess_crop(person_crop, size)
        face_tensor = (
            preprocess_crop(face_crop, size) if face_crop is not None else create_empty_face_input(size)
        )

        outputs = self._session.run(
            None,
            {"faces_input": face_tensor, "body_input": person_tensor},
        )

        return AttributeResult(
            estimated_age=float(np.asarray(outputs[0]).flatten()[0]),
            gender_probability=float(np.asarray(outputs[1]).flatten()[0]),
            category_index=int(np.asarray(outputs[2]).flatten()[0]),
            face_available=face_crop is not None,
        )