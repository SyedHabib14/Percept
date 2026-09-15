"""YOLO-pose inference for PDC's inspected dpose ONNX export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from pdc.analyzer import _resolve_providers
from pdc.config import ModelConfig
from pdc.model_hub import resolve_weight

logger = logging.getLogger("pdc.pose_estimator")

# Discovered by graph inspection and real-image validation on 2026-09-11:
# `images`: float16 [1, 3, 640, 640]; `output0`: float16 [1, 57, 8400].
# Each candidate is [cx, cy, w, h, person, face, 17 * (x, y, visibility)].
# Coordinates are 640-letterbox pixels and the graph contains no NMS.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_face", "right_face",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
CLASS_NAMES = ("person", "face")


@dataclass
class PoseKeypoint:
    name: str
    x: float
    y: float
    visibility: float


@dataclass
class PoseResult:
    keypoints: List[PoseKeypoint]
    box: Tuple[int, int, int, int]
    confidence: float
    class_name: str


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - resized_w) / 2, (size - resized_h) / 2
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, round(pad_y - 0.1), round(pad_y + 0.1), round(pad_x - 0.1),
        round(pad_x + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, pad_x, pad_y


def decode_output(
    output: np.ndarray, image_shape: tuple[int, int], confidence: float, iou: float,
    scale: float, pad_x: float, pad_y: float,
) -> List[PoseResult]:
    """Decode the inspected [1, 57, 8400] contract without model I/O."""
    data = np.asarray(output, dtype=np.float32)
    if data.shape == (1, 57, 8400):
        data = data[0]
    if data.shape != (57, 8400):
        raise ValueError(f"Unexpected dpose output shape {data.shape}; expected [1, 57, 8400].")
    height, width = image_shape
    results: List[PoseResult] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        scores = data[4 + class_index]
        indices = np.where(scores >= confidence)[0]
        if not len(indices):
            continue
        xywh = data[:4, indices].T
        nms_boxes = np.column_stack((xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]))
        kept = cv2.dnn.NMSBoxes(nms_boxes.tolist(), scores[indices].tolist(), confidence, iou)
        for local_index in np.asarray(kept).reshape(-1):
            index = indices[local_index]
            cx, cy, box_w, box_h = data[:4, index]
            x1 = int(np.clip(round((cx - box_w / 2 - pad_x) / scale), 0, width))
            y1 = int(np.clip(round((cy - box_h / 2 - pad_y) / scale), 0, height))
            x2 = int(np.clip(round((cx + box_w / 2 - pad_x) / scale), 0, width))
            y2 = int(np.clip(round((cy + box_h / 2 - pad_y) / scale), 0, height))
            raw_keypoints = data[6:, index].reshape(17, 3)
            keypoints = [
                PoseKeypoint(name, float((x - pad_x) / scale), float((y - pad_y) / scale), float(visibility))
                for name, (x, y, visibility) in zip(KEYPOINT_NAMES, raw_keypoints)
            ]
            results.append(PoseResult(keypoints, (x1, y1, x2, y2), float(scores[index]), class_name))
    return sorted(results, key=lambda item: (item.box[1], item.box[0]))


class PoseEstimator:
    def __init__(self, config: ModelConfig):
        self._config = config
        path = resolve_weight(config.pose_weights, url=config.pose_weights_url, hf_repo_id=config.hf_repo_id)
        self._session = ort.InferenceSession(str(path), providers=_resolve_providers(config.device))
        self._input_name = self._session.get_inputs()[0].name

    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        padded, scale, pad_x, pad_y = letterbox(image)
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float16) / 255.0
        output = self._session.run(None, {self._input_name: tensor})[0]
        return decode_output(output, image.shape[:2], self._config.detection_confidence, self._config.detection_iou, scale, pad_x, pad_y)
