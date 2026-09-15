"""
Person Detector & Classifier (PDC)
===================================

A two-stage computer vision pipeline that detects people and faces in an
image, then estimates per-person visual attributes (age band + gender).

Public API:
    >>> from pdc import PDCPipeline, PDCConfig
    >>> pipeline = PDCPipeline(PDCConfig.load())
    >>> result = pipeline.process_image("photo.jpg")
    >>> result.annotated_image.save("out.jpg")
"""

from pdc.config import PDCConfig, ModelConfig, VisualizationConfig
from pdc.pipeline import PDCPipeline, FrameResult

__all__ = [
    "PDCPipeline",
    "PDCConfig",
    "ModelConfig",
    "VisualizationConfig",
    "FrameResult",
]

__version__ = "1.0.0"
