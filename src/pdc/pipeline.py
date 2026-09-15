"""
The single entry point every inference surface (CLI image/batch tool,
webcam script, and API) is built on top of.

Loading the two models is the expensive part, so `PDCPipeline` loads
them once in `__init__` and reuses them for every subsequent call to
`process_image`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from pdc.analyzer import AttributeAnalyzer
from pdc.association import match_face_to_person
from pdc.config import PDCConfig
from pdc.detector import PersonFaceDetector
from pdc.detector import Detection
from pdc.pose_estimator import PoseEstimator, PoseResult
from pdc.metadata import BodyVisibilityConfig, classify_body_visibility, head_orientation, image_metadata, summarize_camera_orientation
from pdc.age_smoother import AgeSmoother
from pdc.pipeline_types import PersonAnalysis
from pdc.visualization import render_annotations

logger = logging.getLogger("pdc.pipeline")

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class FrameResult:
    """Everything produced by running the pipeline on a single frame/image."""

    persons: list
    faces: list
    analysis_results: List[PersonAnalysis]
    annotated_image: Image.Image
    timing_ms: dict = field(default_factory=dict)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    image_metadata: dict = field(default_factory=dict)

    def as_dict(self, include_metadata: bool = False, clean: bool = False) -> dict:
        result = {
            "person_count": self.person_count,
            "face_count": self.face_count,
            "people": [r.as_dict(include_metadata=include_metadata) for r in self.analysis_results],
        }
        if not clean:
            result["timing_ms"] = self.timing_ms
        if include_metadata:
            result.update({"schema_version": "1.1.0", "image_metadata": self.image_metadata})
        return result


def _load_image(source: ImageInput) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source

    path = Path(source)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


class PDCPipeline:
    """Two-stage person detection + attribute analysis pipeline."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, config: PDCConfig | None = None):
        self.config = config or PDCConfig.load()
        logger.info("Loading PDC pipeline components...")
        # Phase 1: dpose matched bundled detector recall/boxes and was ~11x faster on CPU.
        self.pose_estimator = PoseEstimator(self.config.model)
        self.detector = None
        self.analyzer = AttributeAnalyzer(self.config.model)
        self.age_smoother = AgeSmoother(self.config.visualization.age_smoothing_window)
        self._visibility_history = {}
        logger.info(
            "Pipeline ready. Analysis providers: %s", self.analyzer.active_providers
        )

    def process_image(self, source: ImageInput, *, live: bool = False) -> FrameResult:
        """Run the full pipeline on one image (file path or BGR ndarray)."""
        t0 = time.perf_counter()
        image = _load_image(source)

        t1 = time.perf_counter()
        poses = self.pose_estimator.estimate(image)
        persons, faces = self._split_detections(poses)

        t2 = time.perf_counter()
        analysis_results = self._analyze_people(image, persons, faces, poses, live=live)
        if self.config.visualization.age_smoothing == "rolling_average" and not live:
            logger.warning("age_smoothing is live-only and is ignored for image/batch/API processing.")

        t3 = time.perf_counter()
        annotated = render_annotations(image, analysis_results, self.config.visualization)
        t4 = time.perf_counter()

        return FrameResult(
            persons=persons,
            faces=faces,
            analysis_results=analysis_results,
            annotated_image=annotated,
            timing_ms={
                "load": round((t1 - t0) * 1000, 2),
                "detect": round((t2 - t1) * 1000, 2),
                "analyze": round((t3 - t2) * 1000, 2),
                "render": round((t4 - t3) * 1000, 2),
                "total": round((t4 - t0) * 1000, 2),
            },
            image_metadata=image_metadata(image, self.config.model.min_image_dimension),
        )

    def process_batch(self, sources: List[ImageInput]) -> List[FrameResult]:
        """Run the pipeline sequentially over a list of images."""
        return [self.process_image(source) for source in sources]

    def iter_directory(self, directory: Union[str, Path]) -> List[Path]:
        """Collect supported image paths from a directory (non-recursive)."""
        directory = Path(directory)
        return sorted(
            p for p in directory.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def _split_detections(self, poses: List[PoseResult]):
        persons, faces = [], []
        for pose in poses:
            detection = Detection(pose.box, pose.confidence, pose.class_name)
            (persons if pose.class_name == self.config.model.person_class_name else faces).append(detection)
        return persons, faces

    def _analyze_people(self, image: np.ndarray, persons, faces, poses: List[PoseResult], *, live: bool = False) -> List[PersonAnalysis]:
        results: List[PersonAnalysis] = []
        now = time.monotonic() if live else None

        for person_index, person in enumerate(persons):
            person_id = f"person_{person_index + 1:03d}"
            px1, py1, px2, py2 = person.box
            person_crop = image[max(0, py1) : max(0, py2), max(0, px1) : max(0, px2)]
            if person_crop.size == 0:
                logger.warning("Skipping degenerate person crop at index %d", person_index)
                continue

            best_face = match_face_to_person(person.box, faces)
            face_crop = None
            if best_face is not None:
                fx1, fy1, fx2, fy2 = best_face.box
                candidate = image[max(0, fy1) : max(0, fy2), max(0, fx1) : max(0, fx2)]
                face_crop = candidate if candidate.size > 0 else None

            attributes = self.analyzer.analyze(face_crop=face_crop, person_crop=person_crop)

            # Steps 4 and 5 of the pipeline both start from the same person crop +
            # pose output, but they don't need to run "concurrently" in any real
            # sense: the analyzer call above is the expensive part (an ONNX forward
            # pass, tens of ms), while everything classify_body_visibility() does
            # below is pure math over <=17 points (microseconds). There's nothing
            # to parallelize -- it's already off the critical path by construction.

            # A per-person identity that survives across frames is needed for any
            # live-only temporal smoothing (age rolling-average and the visibility
            # stability vote below). `self.age_smoother` is an IoU tracker plus an
            # age-averaging layer on top; here it is always run in live mode to get
            # the IoU-matched track_id, regardless of whether the age_smoothing
            # *feature* is enabled -- the two are independent uses of one tracker.
            track_id, smoothed_age = None, attributes.estimated_age
            if live:
                face_valid = best_face is not None and attributes.face_available
                track_id, tracked_age = self.age_smoother.update(
                    person.box, attributes.estimated_age, attributes.face_available, face_valid, now)
                if self.config.visualization.age_smoothing == "rolling_average":
                    smoothed_age = tracked_age

            pose = next((item for item in poses if item.class_name == "person" and item.box == person.box), None)
            metadata = None
            if pose is not None:
                visibility = classify_body_visibility(
                    pose.keypoints, person.box, image.shape[:2], pose.confidence,
                    BodyVisibilityConfig(keypoint_visibility=self.config.model.pose_keypoint_visibility))
                if live:
                    # Keyed by the IoU-matched track_id (stable across frames for the
                    # same person), NOT the raw pixel box: the box moves/jitters every
                    # frame even for a stationary subject, so keying on it directly
                    # would silently create a fresh, single-entry history each frame
                    # and this majority-vote smoothing would never engage.
                    history = self._visibility_history.setdefault(track_id, [])
                    history.append(visibility["body_visibility"])
                    del history[:-3]
                    if len(history) >= 2:
                        candidates = {label: history.count(label) for label in set(history)}
                        stable_label, stable_count = max(candidates.items(), key=lambda item: item[1])
                        if stable_count >= 2:
                            visibility["body_visibility"] = stable_label
                code = {"upper_only": "UB", "lower_only": "LB", "full": "FB",
                        "face_only": "FO", "body_only": "BO"}[visibility["body_visibility"]]
                raw_orientation = head_orientation(pose.keypoints, self.config.model.pose_keypoint_visibility)
                metadata = {**visibility, "visibility_class": code,
                            "head_orientation": raw_orientation,
                            "camera_orientation": summarize_camera_orientation(raw_orientation)}
            results.append(
                PersonAnalysis(
                    person_index=person_index,
                    person=person,
                    face=best_face,
                    analysis=attributes,
                    pose=pose,
                    metadata=metadata,
                    track_id=track_id,
                    raw_age=attributes.estimated_age,
                    smoothed_age=smoothed_age,
                    person_id=person_id,
                )
            )

        if live:
            # Prune history for tracks the IoU tracker has since expired, so this
            # dict doesn't grow without bound over a long-running webcam session.
            active_ids = self.age_smoother.active_track_ids()
            self._visibility_history = {
                track_id: history for track_id, history in self._visibility_history.items()
                if track_id in active_ids
            }

        return results