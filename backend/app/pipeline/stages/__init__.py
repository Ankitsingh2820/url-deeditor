"""The de-edit pipeline, in execution order.

Weights approximate real cost so the progress bar is honest rather than linear.
Swap a PlaceholderStage for the real implementation and nothing else moves.
"""

from __future__ import annotations

from app.pipeline.base import Stage
from app.pipeline.stages.export import ExportStage
from app.pipeline.stages.finalize import FinalizeStage
from app.pipeline.stages.ingest import IngestStage
from app.pipeline.stages.inpaint import InpaintStage
from app.pipeline.stages.masks import MaskStage
from app.pipeline.stages.ocr import OcrStage
from app.pipeline.stages.overlays import OverlayStage
from app.pipeline.stages.placeholder import PlaceholderStage
from app.pipeline.stages.probe import ProbeStage
from app.pipeline.stages.sample import SampleStage
from app.pipeline.stages.scenes import SceneStage
from app.pipeline.stages.text_tracks import TextTrackStage


def build_pipeline() -> list[Stage]:
    return [
        # S1-S2 : acquire + normalise
        IngestStage(),
        ProbeStage(),
        # S3-S4 : structure
        SceneStage(),
        SampleStage(),
        # S5-S6 : text + speech
        OcrStage(),
        PlaceholderStage("asr", weight=1.5, required=False, block="G"),
        TextTrackStage(),
        # S7 : image / product pop-ups
        OverlayStage(),
        PlaceholderStage("vlm", weight=1.0, required=False, block="G"),
        # S8-S10 : removal + export
        MaskStage(),
        InpaintStage(),
        ExportStage(),
        # Always last: serialise the editable project.
        FinalizeStage(),
    ]


__all__ = ["build_pipeline"]
