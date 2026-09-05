"""S7b - semantic pass over the scenes.

The computer-vision stages know *that* something is there. This stage is the
only part of the pipeline that knows *what* it is: it names the scene and rules
on each pop-up candidate, turning `kind: "overlay", label: null` into
`kind: "product_popup", label: "serum bottle"` -- and, just as valuable, throws
out the candidates that are really a poster on the wall or a phone in someone's
hand, which geometry cannot distinguish from a pasted graphic.

Cost is bounded on purpose:
  * one call per scene, capped at `vlm_max_scenes`, not one per frame or per box
  * the keyframe and every candidate crop for that scene travel in one request
  * a failed or slow call degrades the job, it never fails it

With no API key the provider factory returns a no-op and this stage records
`vlm_unavailable`; every CV result upstream still stands.
"""

from __future__ import annotations

import logging

import cv2

from app.ai import Candidate, get_provider
from app.config import settings
from app.db.models import JobStatus
from app.pipeline import geometry
from app.pipeline.base import Stage
from app.pipeline.context import JobContext

log = logging.getLogger(__name__)

#: JPEG quality for the crops sent upstream -- small enough to keep the request
#: light, good enough to read a product label
CROP_QUALITY = 82
#: pad each crop so the model sees the asset's border and a little context
CROP_PAD = 0.02


def encode_jpeg(image, quality: int = CROP_QUALITY) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def crop_box(image, box: geometry.Box, pad: float = CROP_PAD):
    height, width = image.shape[:2]
    x, y, w, h = geometry.to_pixels(geometry.dilate(box, pad), width, height)
    return image[y:y + h, x:x + w]


class VlmStage(Stage):
    name = "vlm"
    status = JobStatus.ANALYZING
    weight = 1.0
    required = False

    def skip_if(self, ctx: JobContext) -> str | None:
        if not settings.ai_enabled:
            return "ai_disabled"
        if not ctx.options.ai_enabled:
            return "ai_disabled_for_job"
        if not (ctx.get("scenes") or {}).get("scenes"):
            return "no_scenes"
        return None

    def run(self, ctx: JobContext) -> dict:
        provider = get_provider()
        if not provider.available:
            ctx.degrade(f"vlm_unavailable:{settings.vlm_provider}")
            ctx.log("no VLM credentials; keeping the computer-vision results as they are")
            return {"provider": "noop", "scenes_analysed": 0, "verdicts": {}, "insights": {}}

        ws = ctx.workspace
        scenes = ctx.require("scenes")["scenes"][: settings.vlm_max_scenes]
        overlays = (ctx.get("overlays") or {}).get("tracks", [])
        transcript = (ctx.get("asr") or {}).get("segments", [])

        by_scene: dict[str, list[dict]] = {}
        for track in overlays:
            by_scene.setdefault(track.get("scene_id") or "", []).append(track)

        insights: dict[str, dict] = {}
        verdicts: dict[str, dict] = {}
        failures = 0

        for index, scene in enumerate(scenes):
            ctx.raise_if_cancelled()
            ctx.progress(index / max(1, len(scenes)), f"{provider.name}: {scene['id']}")

            keyframe_path = scene.get("keyframe")
            if not keyframe_path:
                continue
            image = cv2.imread(str(ws.path(keyframe_path)))
            if image is None:
                continue
            keyframe = encode_jpeg(image)
            if keyframe is None:
                continue

            candidates = []
            for track in by_scene.get(scene["id"], []):
                patch = crop_box(image, geometry.from_dict(track["box"]))
                jpeg = encode_jpeg(patch)
                if jpeg:
                    candidates.append(Candidate(id=track["id"], jpeg=jpeg, box=track["box"]))

            analysis = provider.analyse_scene(
                keyframe, candidates,
                context=self._context(scene, transcript),
            )
            if analysis is None:
                failures += 1
                continue

            insights[scene["id"]] = {
                "description": analysis.description,
                "shot_type": analysis.shot_type,
                "tags": analysis.tags,
            }
            for candidate in candidates:
                verdict = analysis.verdict_for(candidate.id)
                if verdict is not None:
                    verdicts[candidate.id] = {
                        "is_overlay": verdict.is_overlay,
                        "kind": verdict.kind,
                        "label": verdict.label,
                        "confidence": verdict.confidence,
                    }

        rejected = self._apply(ctx, overlays, verdicts, insights)
        if failures:
            ctx.degrade(f"vlm_calls_failed:{failures}")
        if len(ctx.require("scenes")["scenes"]) > settings.vlm_max_scenes:
            ctx.degrade(f"vlm_scene_cap:{settings.vlm_max_scenes}")

        ctx.log(
            f"{provider.name}: {len(insights)} scenes described, "
            f"{len(verdicts)} candidates adjudicated, {rejected} rejected as not composited"
        )
        return {
            "provider": provider.name,
            "model": settings.gemini_model if provider.name == "gemini"
            else settings.anthropic_model,
            "scenes_analysed": len(insights),
            "insights": insights,
            "verdicts": verdicts,
            "rejected": rejected,
            "failures": failures,
        }

    @staticmethod
    def _context(scene: dict, transcript: list[dict]) -> str:
        """Give the model the words spoken during this shot -- it disambiguates."""
        from app.pipeline.tracking import spoken_between

        spoken = spoken_between(transcript, scene["t_in"], scene["t_out"])
        parts = [f"Scene runs {scene['t_in']:.1f}s to {scene['t_out']:.1f}s."]
        if spoken:
            parts.append(f'Spoken during this shot: "{spoken[:400]}"')
        return " ".join(parts)

    def _apply(self, ctx: JobContext, overlays: list[dict], verdicts: dict[str, dict],
               insights: dict[str, dict]) -> int:
        """Write the semantics back onto the CV results.

        Rejected candidates are marked, not deleted: the mask stage reads
        `rejected` so a false positive stops being inpainted, while the record
        of what the detector proposed survives for inspection.
        """
        rejected = 0
        for track in overlays:
            verdict = verdicts.get(track["id"])
            if verdict is None:
                continue
            if verdict["is_overlay"]:
                track["kind"] = verdict["kind"]
                track["label"] = verdict["label"]
                track["source"] = "cv+vlm"
                track["vlm_confidence"] = verdict["confidence"]
            else:
                track["rejected"] = True
                track["source"] = "cv+vlm"
                track["vlm_confidence"] = verdict["confidence"]
                rejected += 1

        for scene in ctx.require("scenes")["scenes"]:
            insight = insights.get(scene["id"])
            if insight:
                scene["ai"] = insight
        return rejected
