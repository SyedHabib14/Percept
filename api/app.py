"""
PDC Inference API.

A thin FastAPI wrapper around `PDCPipeline` so the exact same detection
+ attribute-analysis + visualization pipeline used locally can be
deployed as an online service (Docker container, serverless GPU host,
or platform-as-a-service — see README "Deploying Online").

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           liveness + model/device info
    POST /infer            multipart image upload -> structured JSON
                            (optionally + base64 annotated image)
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdc import PDCConfig, PDCPipeline  # noqa: E402
from pdc import __version__ as PDC_VERSION  # noqa: E402

from api.schemas import HealthResponse, InferenceResponse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pdc.api")

_pipeline: PDCPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Loading PDC pipeline for API service...")
    _pipeline = PDCPipeline(PDCConfig.load())
    logger.info("Pipeline loaded. API is ready to serve requests.")
    yield
    _pipeline = None


app = FastAPI(
    title="PDC — Person Detector & Classifier API",
    description="Detect people and faces, then estimate age band + gender per person.",
    version=PDC_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> PDCPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")
    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        version=PDC_VERSION,
        device=pipeline.config.model.device,
        analysis_providers=pipeline.analyzer.active_providers,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(..., description="JPEG/PNG image to analyze"),
    include_image: bool = Query(False, description="Include the annotated image as base64 JPEG"),
) -> InferenceResponse:
    pipeline = get_pipeline()

    raw_bytes = await file.read()
    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    result = pipeline.process_image(image_bgr)

    people = []
    for r in result.analysis_results:
        analysis_payload = None
        if r.analysis is not None:
            gender_label = pipeline.config.visualization.gender_labels.get(
                r.analysis.category_index, f"Class {r.analysis.category_index}"
            )
            from pdc.visualization import get_age_bin

            age_band = get_age_bin(r.analysis.estimated_age, pipeline.config.visualization.age_bins)
            analysis_payload = {
                **r.analysis.as_dict(),
                "gender_label": gender_label,
                "age_band": age_band,
            }

        people.append(
            {
                "person_id": r.person_id,
                "person_index": r.person_index,
                "person": r.person.as_dict(),
                "face": r.face.as_dict() if r.face is not None else None,
                "analysis": analysis_payload,
                "visibility_class": (r.metadata or {}).get("visibility_class", "BO"),
                "body_visibility": (r.metadata or {}).get("body_visibility", "body_only"),
                "visibility_scores": (r.metadata or {}).get("scores", {}),
                "visibility_evidence": (r.metadata or {}).get("evidence", {}),
                "camera_orientation": (r.metadata or {}).get("camera_orientation", "insufficient_evidence"),
            }
        )

    response = InferenceResponse(
        person_count=result.person_count,
        face_count=result.face_count,
        people=people,
        timing_ms=result.timing_ms,
    )

    if include_image:
        buffer = io.BytesIO()
        result.annotated_image.save(buffer, format="JPEG", quality=92)
        response.annotated_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return response