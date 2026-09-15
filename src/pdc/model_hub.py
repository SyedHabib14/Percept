"""
Model weight resolution.

Makes the pipeline runnable on a fresh machine with zero manual setup:
if a configured weight file isn't present locally, it is fetched from
a direct URL or a Hugging Face Hub repo and cached under `models/`.

Resolution order for each weight:
    1. Local path exists on disk           -> use as-is
    2. `hf_repo_id` is configured           -> download via huggingface_hub
    3. A direct `*_weights_url` is set      -> stream download over HTTP
    4. Otherwise                            -> raise a clear, actionable error
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger("pdc.model_hub")


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required weight file cannot be located or fetched."""


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    logger.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    tmp_path.rename(destination)


def _download_from_hf(repo_id: str, filename: str, destination: Path) -> Path:
    try:
        from huggingface_hub import EntryNotFoundError, hf_hub_download
    except ImportError as exc:
        raise ModelNotFoundError(
            "huggingface_hub is not installed but `hf_repo_id` was set. "
            "Install it with `pip install huggingface_hub` or provide a "
            "direct download URL instead."
        ) from exc

    logger.info("Fetching %s from Hugging Face repo %s", filename, repo_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=destination.parent,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    if filename.endswith(".onnx"):
        sidecar = f"{filename}.data"
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=sidecar,
                local_dir=destination.parent,
                token=os.environ.get("HF_TOKEN") or None,
            )
            logger.info("Fetched ONNX external data %s", sidecar)
        except EntryNotFoundError:
            pass
    return cached_path


def resolve_weight(
    local_path: str | Path,
    *,
    url: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
) -> Path:
    """
    Ensure `local_path` exists on disk, downloading it if necessary.
    Returns the resolved, existing Path.
    """
    path = Path(local_path)
    if path.exists():
        return path

    if hf_repo_id:
        return _download_from_hf(hf_repo_id, path.name, path)

    if url:
        _download_with_progress(url, path)
        return path

    raise ModelNotFoundError(
        f"Model weight '{path}' was not found and no `hf_repo_id` or "
        f"download URL is configured. Place the file at that path, or "
        f"set the corresponding *_weights_url / hf_repo_id in your config "
        f"(see config/default.yaml and models/README.md)."
    )
