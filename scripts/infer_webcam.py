#!/usr/bin/env python3
"""
Live webcam / video-stream inference — same detection, analysis, and
"clean visualization" aesthetic as the image pipeline, running in
real time.

Controls
--------
    q / Esc   quit
    s         save a snapshot of the current annotated frame
    space     pause / resume

Examples
--------
    python scripts/infer_webcam.py                        # default camera
    python scripts/infer_webcam.py --source 1              # second camera
    python scripts/infer_webcam.py --source clip.mp4        # a video file
    python scripts/infer_webcam.py --record session.mp4     # save the output
    python scripts/infer_webcam.py --skip-frames 2           # run analysis every 3rd frame
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdc import PDCConfig, PDCPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pdc.cli.webcam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDC live webcam inference")
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index (0, 1, ...), video file path, or stream URL. "
        "Defaults to the value in config/default.yaml.",
    )
    parser.add_argument("--config", default=None, help="Path to a custom YAML config")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--pose-model", default=None, help="Custom pose model path (age/gender model remains fixed)")
    parser.add_argument("--detect-model", default=None, help="Custom detection model path (age/gender model remains fixed)")
    parser.add_argument("--age-display", choices=["exact", "binned"], default=None)
    parser.add_argument("--age-smoothing", choices=["none", "rolling_average"], default=None)
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Reuse the last annotation for N frames between analyses, for higher FPS "
        "on slower hardware.",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Disable the selfie-view mirror flip")
    parser.add_argument("--record", default=None, help="Save the annotated stream to this video file")
    parser.add_argument(
        "--snapshot-dir", default="results/webcam_snapshots", help="Where 's' key saves snapshots"
    )
    return parser.parse_args()


def _resolve_source(raw: str):
    return int(raw) if raw.isdigit() else raw


def _draw_hud(frame_bgr: np.ndarray, fps: float, person_count: int, paused: bool) -> np.ndarray:
    overlay = frame_bgr.copy()
    hud_height = 34
    cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], hud_height), (10, 14, 22), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)

    status = "PAUSED" if paused else f"{fps:4.1f} FPS"
    text = f"LIVE  |  {status}  |  people: {person_count}  |  q: quit  s: snapshot  space: pause"
    cv2.putText(
        frame_bgr, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame_bgr


def main() -> None:
    args = parse_args()
    config = PDCConfig.load(args.config)
    if args.device:
        config.model.device = args.device
    if args.pose_model:
        config.model.pose_weights = args.pose_model
    if args.detect_model:
        config.model.detection_weights = args.detect_model
    if args.age_display:
        config.visualization.age_display_mode = args.age_display
    if args.age_smoothing:
        config.visualization.age_smoothing = args.age_smoothing
    if args.source:
        config.webcam.source = args.source
    if args.skip_frames is not None:
        config.webcam.process_every_n_frames = max(1, args.skip_frames + 1)
    if args.no_mirror:
        config.webcam.mirror = False
    if args.record:
        config.webcam.record_output = args.record

    pipeline = PDCPipeline(config)

    source = _resolve_source(config.webcam.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        logger.error("Could not open video source: %s", source)
        sys.exit(1)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    input_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if config.webcam.record_output:
        Path(config.webcam.record_output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            config.webcam.record_output, fourcc, input_fps, (frame_width, frame_height)
        )

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting live inference. Press 'q' to quit.")

    frame_index = 0
    last_annotated_bgr = None
    last_person_count = 0
    paused = False
    fps_smoothed = 0.0

    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    logger.info("Video source ended.")
                    break

                if config.webcam.mirror and isinstance(source, int):
                    frame = cv2.flip(frame, 1)

                t0 = time.perf_counter()
                if frame_index % config.webcam.process_every_n_frames == 0:
                    result = pipeline.process_image(frame, live=True)
                    annotated_rgb = np.array(result.annotated_image)
                    last_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                    last_person_count = result.person_count
                else:
                    last_annotated_bgr = frame if last_annotated_bgr is None else last_annotated_bgr
                elapsed = max(time.perf_counter() - t0, 1e-6)

                instantaneous_fps = 1.0 / elapsed
                fps_smoothed = (
                    instantaneous_fps
                    if fps_smoothed == 0.0
                    else (0.9 * fps_smoothed + 0.1 * instantaneous_fps)
                )
                frame_index += 1

            display_frame = last_annotated_bgr
            if display_frame is None:
                continue

            if config.webcam.show_fps:
                display_frame = _draw_hud(display_frame, fps_smoothed, last_person_count, paused)

            if writer is not None and not paused:
                writer.write(cv2.resize(display_frame, (frame_width, frame_height)))

            cv2.imshow(config.webcam.window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or Esc
                break
            elif key == ord("s"):
                snapshot_path = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snapshot_path), display_frame)
                logger.info("Saved snapshot: %s", snapshot_path)
            elif key == ord(" "):
                paused = not paused

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Session ended.")


if __name__ == "__main__":
    main()
