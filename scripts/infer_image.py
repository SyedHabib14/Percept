#!/usr/bin/env python3
"""
Image inference CLI — single file, a directory of images (batch), or a
glob pattern. Produces annotated images and, optionally, a JSON report.

Examples
--------
    # Single image, view + save next to a `results/` folder
    python scripts/infer_image.py --input photo.jpg

    # Batch: every image in a folder
    python scripts/infer_image.py --input ./my_photos --output-dir ./results

    # Batch with a structured JSON report and no preview window
    python scripts/infer_image.py --input ./my_photos --json report.json --no-display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdc import PDCConfig, PDCPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pdc.cli.image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDC image / batch inference")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save annotated images (default: ./results)",
    )
    parser.add_argument("--config", default=None, help="Path to a custom YAML config")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--pose-model", default=None, help="Custom pose model path (age/gender model remains fixed)")
    parser.add_argument("--detect-model", default=None, help="Custom detection model path (age/gender model remains fixed)")
    parser.add_argument("--json", default=None, help="Write a structured JSON report to this path")
    parser.add_argument("--json-output", choices=["none", "full", "clean"], default=None,
                        help="JSON mode: none (default), full (metadata/timing), or clean (essential fields).")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip opening a preview window (useful on headless servers)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PDCConfig.load(args.config)
    if args.device:
        config.model.device = args.device
    if args.pose_model:
        config.model.pose_weights = args.pose_model
    if args.detect_model:
        config.model.detection_weights = args.detect_model

    json_mode = args.json_output
    if json_mode is None:
        if sys.stdin.isatty():
            json_mode = "clean" if input("Output clean JSON? (y/n) ").strip().lower() in {"y", "yes"} else "none"
        else:
            json_mode = "none"
            logger.warning("No --json-output supplied in a non-interactive session; defaulting to none.")

    pipeline = PDCPipeline(config)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = pipeline.iter_directory(input_path)
        if not image_paths:
            logger.error("No supported images found in %s", input_path)
            sys.exit(1)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Running inference on %d image(s)...", len(image_paths))
    report = []

    for path in image_paths:
        try:
            result = pipeline.process_image(path)
        except Exception:
            logger.exception("Failed to process %s — skipping.", path)
            continue

        out_path = output_dir / f"{path.stem}_annotated.jpg"
        result.annotated_image.save(out_path, quality=95)

        logger.info(
            "%-30s persons=%d faces=%d total=%.1fms -> %s",
            path.name,
            result.person_count,
            result.face_count,
            result.timing_ms["total"],
            out_path,
        )

        if json_mode != "none":
            report.append({"source": str(path), "output": str(out_path), **result.as_dict(
                include_metadata=json_mode == "full", clean=json_mode == "clean")})

        if not args.no_display and len(image_paths) == 1:
            _show_preview(result.annotated_image)

    if json_mode != "none":
        json_path = Path(args.json) if args.json else output_dir / "report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Wrote %s JSON report to %s", json_mode, json_path)

    logger.info("Done. %d/%d image(s) processed successfully.", len(report), len(image_paths))


def _show_preview(pil_image) -> None:
    try:
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("PDC — Result (press any key to close)", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        logger.info("Preview window unavailable in this environment; image was saved to disk.")


if __name__ == "__main__":
    main()
