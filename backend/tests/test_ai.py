"""The AI layer: provider selection, response parsing, and the semantic pass.

The VLM stage is exercised end to end with a stub provider, so the adjudication
path -- naming a pop-up, rejecting a false positive, and keeping the rejected one
out of the inpainting mask -- is covered without an API key or a network call.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.base import RESPONSE_SCHEMA, Candidate, OverlayVerdict, SceneAnalysis
from app.pipeline.context import JobContext
from app.pipeline.stages.masks import MaskStage
from app.pipeline.stages.vlm import VlmStage, crop_box, encode_jpeg
from app.schemas.job import JobOptions
from app.storage import Workspace

# ------------------------------------------------------------- response parsing


def test_scene_analysis_parses_a_well_formed_payload():
    analysis = SceneAnalysis.from_payload({
        "description": "Woman holds a serum bottle to camera",
        "shot_type": "medium close-up",
        "tags": ["ugc", "talking head"],
        "overlays": [
            {"id": "ov_001", "is_overlay": True, "kind": "product_popup",
             "label": "serum bottle", "confidence": 0.87},
        ],
    })
    assert analysis.description.startswith("Woman holds")
    assert analysis.tags == ["ugc", "talking head"]
    verdict = analysis.verdict_for("ov_001")
    assert verdict and verdict.kind == "product_popup" and verdict.label == "serum bottle"
    assert analysis.verdict_for("ov_999") is None


def test_scene_analysis_survives_a_sparse_or_odd_payload():
    """A model can return junk; parsing must never raise into the pipeline."""
    analysis = SceneAnalysis.from_payload({})
    assert analysis.description == "" and analysis.overlays == []

    odd = SceneAnalysis.from_payload({
        "description": None,
        "tags": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "overlays": [{"id": "ov_001", "is_overlay": True, "kind": "not-a-real-kind",
                      "label": "   ", "confidence": None}],
    })
    assert len(odd.tags) == 6                      # capped
    assert odd.overlays[0].kind == "none"          # unknown kind rejected
    assert odd.overlays[0].label is None           # blank label normalised
    assert odd.overlays[0].confidence == 0.0


def test_response_schema_is_strict_enough_to_be_useful():
    """Both providers constrain output to this; loose schemas defeat the point."""
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    item = RESPONSE_SCHEMA["properties"]["overlays"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "is_overlay", "kind", "label", "confidence"}
    assert "none" in item["properties"]["kind"]["enum"]


# ---------------------------------------------------------- provider selection


def test_provider_is_noop_when_ai_is_disabled(monkeypatch):
    from app import ai

    monkeypatch.setattr(ai.settings, "ai_enabled", False)
    assert ai.get_provider().available is False


def test_provider_is_noop_when_the_key_is_missing(monkeypatch):
    from app import ai

    monkeypatch.setattr(ai.settings, "ai_enabled", True)
    monkeypatch.setattr(ai.settings, "vlm_provider", "gemini")
    monkeypatch.setattr(ai.settings, "gemini_api_key", None)
    provider = ai.get_provider()
    assert provider.name == "noop" and provider.available is False


def test_provider_selection_reports_unavailable_rather_than_raising(monkeypatch):
    """A missing key must never be an exception -- it degrades the job instead."""
    from app import ai

    monkeypatch.setattr(ai.settings, "ai_enabled", True)
    monkeypatch.setattr(ai.settings, "anthropic_api_key", None)
    assert ai.get_provider("claude").available is False
    assert ai.get_provider("nonsense").available is False


def test_noop_provider_returns_none():
    from app.ai.providers.noop import NoopVLM

    assert NoopVLM().analyse_scene(b"", []) is None


# ------------------------------------------------------------------ image prep


def test_encode_jpeg_and_crop_round_trip():
    image = np.zeros((200, 100, 3), np.uint8)
    cv2.rectangle(image, (50, 20), (90, 60), (255, 255, 255), -1)

    patch = crop_box(image, (0.5, 0.1, 0.4, 0.2))
    assert patch.size > 0
    jpeg = encode_jpeg(patch)
    assert jpeg and jpeg[:2] == b"\xff\xd8"        # JPEG magic


# ------------------------------------------------------------ the semantic pass


class StubVLM:
    """Deterministic stand-in: names one candidate, rejects the other."""

    name = "stub"
    available = True

    def __init__(self):
        self.calls: list[int] = []

    def analyse_scene(self, keyframe_jpeg, candidates, *, context=""):
        self.calls.append(len(candidates))
        self.last_context = context
        return SceneAnalysis(
            description="Woman holding a serum bottle in a bathroom",
            shot_type="medium close-up",
            tags=["ugc"],
            overlays=[
                OverlayVerdict(id="ov_001", is_overlay=True, kind="product_popup",
                               label="serum bottle", confidence=0.9),
                OverlayVerdict(id="ov_002", is_overlay=False, kind="none",
                               label=None, confidence=0.8),
            ],
        )


@pytest.fixture
def vlm_ctx(tmp_path: Path, monkeypatch) -> JobContext:
    monkeypatch.setattr("app.pipeline.context.emit", lambda *a, **k: None)
    workspace = Workspace(job_id="job_vlm", root=tmp_path).ensure()

    keyframe = workspace.path("frames", "sc_001_key.jpg")
    cv2.imwrite(str(keyframe), np.full((400, 220, 3), 90, np.uint8))

    ctx = JobContext(job_id="job_vlm", workspace=workspace, options=JobOptions(),
                     source_kind="upload", source_ref="clip.mp4")
    ctx.set("probe", {"duration": 6.0, "width": 220, "height": 400, "fps": 30})
    ctx.set("scenes", {"scenes": [{
        "id": "sc_001", "index": 0, "t_in": 0.0, "t_out": 6.0, "duration": 6.0,
        "keyframe": "frames/sc_001_key.jpg", "clip": None, "clean_clip": None, "ai": None,
    }]})
    ctx.set("overlays", {"tracks": [
        {"id": "ov_001", "kind": "overlay", "label": None, "t_in": 1.0, "t_out": 3.0,
         "scene_id": "sc_001", "box": {"x": 0.5, "y": 0.1, "w": 0.35, "h": 0.25},
         "confidence": 0.6, "source": "cv"},
        {"id": "ov_002", "kind": "overlay", "label": None, "t_in": 4.0, "t_out": 5.0,
         "scene_id": "sc_001", "box": {"x": 0.05, "y": 0.6, "w": 0.25, "h": 0.2},
         "confidence": 0.6, "source": "cv"},
    ]})
    ctx.set("asr", {"segments": [{"t_in": 0.5, "t_out": 3.0, "text": "look at this serum"}]})
    return ctx


def test_vlm_names_a_real_popup_and_rejects_a_false_positive(vlm_ctx, monkeypatch):
    stub = StubVLM()
    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: stub)

    result = VlmStage().run(vlm_ctx)

    assert result["scenes_analysed"] == 1
    assert result["rejected"] == 1
    # One call for the scene, carrying both candidates -- not one call per box.
    assert stub.calls == [2]

    tracks = {t["id"]: t for t in vlm_ctx.get("overlays")["tracks"]}
    assert tracks["ov_001"]["kind"] == "product_popup"
    assert tracks["ov_001"]["label"] == "serum bottle"
    assert tracks["ov_001"]["source"] == "cv+vlm"
    assert not tracks["ov_001"].get("rejected")

    assert tracks["ov_002"]["rejected"] is True     # marked, not deleted
    assert tracks["ov_002"]["label"] is None


def test_vlm_attaches_scene_descriptions(vlm_ctx, monkeypatch):
    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: StubVLM())
    VlmStage().run(vlm_ctx)

    scene = vlm_ctx.get("scenes")["scenes"][0]
    assert scene["ai"]["description"].startswith("Woman holding")
    assert scene["ai"]["shot_type"] == "medium close-up"


def test_vlm_passes_the_spoken_words_as_context(vlm_ctx, monkeypatch):
    """The transcript disambiguates what the model is looking at."""
    stub = StubVLM()
    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: stub)
    VlmStage().run(vlm_ctx)
    assert "look at this serum" in stub.last_context


def test_a_rejected_candidate_is_not_inpainted(vlm_ctx, monkeypatch):
    """The payoff: a false positive stops being painted out of the footage."""
    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: StubVLM())
    VlmStage().run(vlm_ctx)

    masks = MaskStage().run(vlm_ctx)
    masked = {box["source_id"] for seg in masks["segments"] for box in seg["boxes"]}
    assert masked == {"ov_001"}


def test_vlm_without_credentials_degrades_and_changes_nothing(vlm_ctx, monkeypatch):
    from app.ai.providers.noop import NoopVLM

    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: NoopVLM())
    result = VlmStage().run(vlm_ctx)

    assert result["provider"] == "noop"
    assert result["scenes_analysed"] == 0
    assert any("vlm_unavailable" in d for d in vlm_ctx.degradations)
    # The CV results survive untouched.
    assert vlm_ctx.get("overlays")["tracks"][0]["kind"] == "overlay"


def test_a_failing_provider_degrades_rather_than_raising(vlm_ctx, monkeypatch):
    class Broken:
        name = "broken"
        available = True

        def analyse_scene(self, *a, **k):
            return None                      # e.g. a timeout or a refusal

    monkeypatch.setattr("app.pipeline.stages.vlm.get_provider", lambda: Broken())
    result = VlmStage().run(vlm_ctx)

    assert result["failures"] == 1
    assert any("vlm_calls_failed" in d for d in vlm_ctx.degradations)


def test_candidate_carries_the_crop_it_was_asked_about():
    candidate = Candidate(id="ov_001", jpeg=b"\xff\xd8", box={"x": 0.1, "y": 0.1,
                                                             "w": 0.2, "h": 0.2})
    assert candidate.id == "ov_001" and candidate.jpeg.startswith(b"\xff\xd8")
