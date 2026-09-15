#!/usr/bin/env python3
"""
Pre-fetch model weights defined in the active config.

Run this once on a fresh machine (or as a Docker build step) so the
first real inference call doesn't pay a download penalty:

    python scripts/download_models.py
    python scripts/download_models.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdc.config import PDCConfig  # noqa: E402
from pdc.model_hub import resolve_weight  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pdc.cli.download")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch PDC model weights")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = PDCConfig.load(args.config).model

    detection_path = resolve_weight(
        config.detection_weights, url=config.detection_weights_url, hf_repo_id=config.hf_repo_id
    )
    logger.info("Detection weights ready: %s", detection_path)

    analysis_path = resolve_weight(
        config.analysis_weights, url=config.analysis_weights_url, hf_repo_id=config.hf_repo_id
    )
    logger.info("Analysis weights ready: %s", analysis_path)

    logger.info("All model weights are present. You're ready to run inference.")


if __name__ == "__main__":
    main()
