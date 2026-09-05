"""The de-edit pipeline, in execution order.

Weights approximate real cost so the progress bar is honest rather than linear.
Ordering carries meaning: ASR runs before text tracking because the transcript is
what classifies a caption, and the VLM runs after overlay detection because it
adjudicates that stage's candidates.
"""

from __future__ import annotations

from app.pipeline.base import Stage
from app.pipeline.stages.asr import AsrStage
from app.pipeline.stages.export import ExportStage
from app.pipeline.stages.finalize import FinalizeStage
from app.pipeline.stages.ingest import IngestStage
from app.pipeline.stages.inpaint import InpaintStage
from app.pipeline.stages.masks import MaskStage
from app.pipeline.stages.ocr import OcrStage
from app.pipeline.stages.overlays import OverlayStage
from app.pipeline.stages.probe import ProbeStage
from app.pipeline.stages.sample import SampleStage
from app.pipeline.stages.scenes import SceneStage
from app.pipeline.stages.text_tracks import TextTrackStage
from app.pipeline.stages.vlm import VlmStage


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
        AsrStage(),
        TextTrackStage(),
        # S7 : image / product pop-ups
        OverlayStage(),
        VlmStage(),
        # S8-S10 : removal + export
        MaskStage(),
        InpaintStage(),
        ExportStage(),
        # Always last: serialise the editable project.
        FinalizeStage(),
    ]


__all__ = ["build_pipeline"]
