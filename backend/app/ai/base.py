"""Provider-agnostic vision-language interface.

The pipeline asks one question per scene and does not care who answers it.
That matters for three reasons: the demo must run with no key at all, the
provider is a cost decision that should not be baked into pipeline code, and
tests need a deterministic stand-in.

**One call per scene, not per frame or per candidate.** A 30 s video has ~900
frames and maybe a dozen scenes; the scene's keyframe plus its candidate crops
go up together and come back as one structured answer. Sending every frame would
be slow and expensive, and sending nothing loses all semantics -- which is
exactly the division of labour here: cheap CV proposes candidates, the model
adjudicates what they *are*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: what a candidate region can turn out to be
OVERLAY_KINDS = ("product_popup", "sticker", "logo", "ui_element", "screenshot", "none")


@dataclass
class OverlayVerdict:
    """The model's ruling on one CV candidate."""

    id: str
    is_overlay: bool
    kind: str = "none"
    label: str | None = None
    confidence: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict) -> OverlayVerdict:
        kind = str(payload.get("kind") or "none")
        label = (payload.get("label") or "").strip() or None
        return cls(
            id=str(payload.get("id") or ""),
            is_overlay=bool(payload.get("is_overlay")),
            kind=kind if kind in OVERLAY_KINDS else "none",
            label=label,
            confidence=float(payload.get("confidence") or 0.0),
        )


@dataclass
class SceneAnalysis:
    """Everything the model returns about one scene."""

    description: str = ""
    shot_type: str | None = None
    tags: list[str] = field(default_factory=list)
    overlays: list[OverlayVerdict] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict) -> SceneAnalysis:
        return cls(
            description=str(payload.get("description") or "").strip(),
            shot_type=(str(payload.get("shot_type") or "").strip() or None),
            tags=[str(t) for t in (payload.get("tags") or [])][:6],
            overlays=[OverlayVerdict.from_payload(o) for o in (payload.get("overlays") or [])],
        )

    def verdict_for(self, candidate_id: str) -> OverlayVerdict | None:
        return next((o for o in self.overlays if o.id == candidate_id), None)


@dataclass
class Candidate:
    """A CV-proposed region, with the crop the model should look at."""

    id: str
    jpeg: bytes
    box: dict[str, float]


@runtime_checkable
class VLMProvider(Protocol):
    name: str

    @property
    def available(self) -> bool:
        """False when the provider has no credentials or no SDK installed."""

    def analyse_scene(
        self,
        keyframe_jpeg: bytes,
        candidates: list[Candidate],
        *,
        context: str = "",
    ) -> SceneAnalysis | None:
        """Describe the scene and rule on each candidate. None on failure."""


#: JSON Schema both providers constrain their output to, so the pipeline gets
#: the same shape back regardless of who answered.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "One sentence describing what happens in this shot.",
        },
        "shot_type": {
            "type": "string",
            "description": "e.g. close-up, medium shot, wide shot, product shot, screen recording",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 5 short tags, e.g. ugc, talking head, unboxing, bathroom",
        },
        "overlays": {
            "type": "array",
            "description": "One entry per candidate region, keyed by the given id.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "is_overlay": {
                        "type": "boolean",
                        "description": (
                            "True only if this region is a graphic composited on top of the "
                            "footage in editing, not something physically in the scene."
                        ),
                    },
                    "kind": {"type": "string", "enum": list(OVERLAY_KINDS)},
                    "label": {
                        "type": "string",
                        "description": (
                            "What the graphic shows, e.g. 'serum bottle'. Empty if none."
                        ),
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["id", "is_overlay", "kind", "label", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["description", "shot_type", "tags", "overlays"],
    "additionalProperties": False,
}
