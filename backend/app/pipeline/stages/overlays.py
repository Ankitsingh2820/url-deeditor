"""S7 - image / product pop-up detection.

Runs per scene, because every signal is relative to the shot it lives in: what
counts as "still" depends on how much that shot moves, and a pop-up's lifetime
is bounded by its scene.

The output slots into the same `overlay_tracks[]` contract the mask, inpaint and
export stages already consume, so detected pop-ups are removed from the clean
plate with no further wiring.

This stage is `required=False`: overlay detection is the most heuristic part of
the pipeline, and a failure here must degrade the result rather than lose the
scenes, captions and clean video that already succeeded.
"""

from __future__ import annotations

import logging

import cv2

from app.db.models import JobStatus
from app.pipeline import geometry, overlays
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.pipeline.overlays import Candidate

log = logging.getLogger(__name__)

#: fewer samples than this and the temporal signals are meaningless
MIN_SAMPLES_PER_SCENE = 4


class OverlayStage(Stage):
    name = "overlays"
    status = JobStatus.ANALYZING
    weight = 1.5
    required = False

    def skip_if(self, ctx: JobContext) -> str | None:
        if not (ctx.get("sample") or {}).get("samples"):
            return "no_sampled_frames"
        return None

    def run(self, ctx: JobContext) -> dict:
        scenes = ctx.require("scenes")["scenes"]
        samples = ctx.require("sample")["samples"]

        # Text is already handled; a caption must not also become an "overlay".
        text_boxes = [
            geometry.from_dict(track["box"])
            for track in (ctx.get("text_tracks") or {}).get("tracks", [])
        ]

        by_scene: dict[str, list[dict]] = {}
        for sample in samples:
            if sample["scene_id"]:
                by_scene.setdefault(sample["scene_id"], []).append(sample)

        found: list[Candidate] = []
        skipped = 0
        for index, scene in enumerate(scenes):
            ctx.raise_if_cancelled()
            ctx.progress(index / max(1, len(scenes)), f"pop-ups: {scene['id']}")
            scene_samples = sorted(by_scene.get(scene["id"], []), key=lambda s: s["t"])
            if len(scene_samples) < MIN_SAMPLES_PER_SCENE:
                skipped += 1
                continue
            found.extend(self._scan_scene(ctx, scene["id"], scene_samples, text_boxes))

        kept = overlays.deduplicate(found)
        if skipped:
            ctx.degrade(f"overlay_scan_skipped_short_scenes:{skipped}")
        if not kept:
            ctx.degrade("no_overlays_detected")

        tracks = [
            {
                "id": f"ov_{i + 1:03d}",
                # Geometry alone cannot say *what* this is; the VLM stage
                # refines kind/label when it is enabled.
                "kind": "overlay",
                "label": None,
                "t_in": round(candidate.t_in, 3),
                "t_out": round(candidate.t_out, 3),
                "scene_id": candidate.scene_id,
                "box": geometry.to_dict(candidate.box),
                "asset": None,
                "confidence": candidate.score,
                "signals": {k: round(v, 4) for k, v in candidate.scores.items()},
                "source": "cv",
            }
            for i, candidate in enumerate(sorted(kept, key=lambda c: c.t_in))
        ]

        ctx.log(f"{len(tracks)} overlay candidates from {len(found)} raw detections")
        return {"tracks": tracks, "count": len(tracks), "raw": len(found)}

    def _scan_scene(self, ctx: JobContext, scene_id: str, samples: list[dict],
                    text_boxes: list[geometry.Box]) -> list[Candidate]:
        frames = []
        times = []
        for sample in samples:
            image = cv2.imread(str(ctx.workspace.path(sample["path"])), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            frames.append(image)
            times.append(sample["t"])
        if len(frames) < MIN_SAMPLES_PER_SCENE:
            return []

        motion = overlays.camera_motion(frames)
        moving = motion >= overlays.MOTION_THRESHOLD_PX
        ctx.log(
            f"{scene_id}: camera {motion:.1f}px -> "
            f"{'stability' if moving else 'transience'} signal"
        )

        candidates: list[Candidate] = []
        for window in overlays.windows(len(frames)):
            slice_ = [frames[i] for i in window]
            if len(slice_) < 2:
                continue
            middle = window[len(window) // 2]
            edges = overlays.edge_map(frames[middle])

            # Exactly one signal is meaningful, and which one depends on whether
            # the camera moved; see the module docstring for why ORing them is
            # actively harmful.
            std_map = overlays.stability_map(slice_)
            counts = overlays.transition_count(slice_)
            raw = (
                overlays.stability_mask(std_map)
                if moving
                else overlays.transience_mask(counts)
            )

            # Restrict to regions that actually contain something, then fill the
            # flat interior of whatever survived.
            mask = cv2.bitwise_and(raw, overlays.structure_mask(frames[middle]))
            mask = overlays.solidify(overlays.clean_mask(mask))

            for box, solidity in overlays.boxes_from_mask(mask):
                if overlays.overlaps_text(box, text_boxes):
                    continue
                detail = overlays.edge_density(edges, box)
                if detail < overlays.MIN_EDGE_DENSITY:
                    continue

                scores = {
                    "stability": overlays.stability_score(std_map, box) if moving else 0.0,
                    "transience": overlays.transience_score(counts, box) if not moving else 0.0,
                    "detail": float(min(1.0, detail / 0.12)),
                    "solidity": solidity,
                }
                candidate = Candidate(
                    box=box, scene_id=scene_id,
                    t_in=times[window[0]], t_out=times[window[-1]], scores=scores,
                )
                if candidate.score < overlays.SCORE_THRESHOLD:
                    continue

                # An overlay has duration. Matching the patch back across the
                # scene both recovers its true lifetime and rejects the
                # single-frame flickers that the spatial scores cannot tell
                # apart from a real asset.
                presence = overlays.patch_presence(frames, middle, box)
                if overlays.present_count(presence) < overlays.MIN_PRESENT_SAMPLES:
                    continue
                candidate.t_in, candidate.t_out = overlays.presence_span(
                    times, presence, reference_index=middle
                )
                candidates.append(candidate)

        return candidates
