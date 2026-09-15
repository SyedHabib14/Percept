"""
Stage 1 — Person & Face Detection.

Thin, production wrapper around the Ultralytics YOLO detector used in
the research notebook. Behavior (thresholds, sorting, class split) is
preserved exactly; the difference is that the model loads once and is
reused across every image, batch item, or webcam frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pdc.config import ModelConfig
from pdc.model_hub import resolve_weight

logger = logging.getLogger("pdc.detector")


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    confidence: float
    class_name: str

    def as_dict(self) -> dict:
        return {"box": self.box, "confidence": self.confidence, "class_name": self.class_name}


class PersonFaceDetector:
    """Loads the detection model once and exposes a simple `.detect()` call."""

    def __init__(self, config: ModelConfig):
        self._config = config

        from ultralytics import YOLO  # local import: keeps import light for API-only use

        weights_path = resolve_weight(
            config.detection_weights,
            url=config.detection_weights_url,
            hf_repo_id=config.hf_repo_id,
        )
        logger.info("Loading detection model from %s", weights_path)
        self._model = YOLO(str(weights_path))

        if config.device == "cuda":
            self._model.to("cuda")

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """
        Run detection on a single BGR image (as returned by cv2.imread).

        Returns:
            persons: detections sorted top-to-bottom, left-to-right
            faces:   detections sorted top-to-bottom, left-to-right
        """
        results = self._model.predict(
            source=image,
            conf=self._config.detection_confidence,
            iou=self._config.detection_iou,
            device=0 if self._config.device == "cuda" else "cpu",
            verbose=False,
        )
        result = results[0]

        persons: List[Detection] = []
        faces: List[Detection] = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = result.names[class_id].lower().strip()

            detection = Detection(
                box=(int(x1), int(y1), int(x2), int(y2)),
                confidence=confidence,
                class_name=class_name,
            )

            if class_name == self._config.person_class_name:
                persons.append(detection)
            elif class_name == self._config.face_class_name:
                faces.append(detection)

        persons.sort(key=lambda d: (d.box[1], d.box[0]))
        faces.sort(key=lambda d: (d.box[1], d.box[0]))

        return persons, faces
