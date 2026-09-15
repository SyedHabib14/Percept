# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

# Runtime libs needed by opencv-python-headless + font for badge rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY scripts/download_models.py ./scripts/download_models.py
COPY pyproject.toml .

ENV PYTHONPATH=/app/src
ENV PDC_DEVICE=cpu

# Bake weights into the image at build time if URLs/HF repo are provided
# via --build-arg, so cold starts don't pay a download penalty. Weights
# placed in ./models before building are copied in either way.
COPY models/ ./models/
ARG PDC_HF_REPO_ID=""
ENV PDC_HF_REPO_ID=${PDC_HF_REPO_ID}
RUN python scripts/download_models.py || echo "Weights not pre-fetched at build time; will fetch on first request."

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
