"""The provider used when no key is configured -- and in tests.

Returning None (rather than raising, or inventing an answer) is what lets the
whole pipeline run without credentials: the VLM stage records a degradation and
the CV results stand on their own.
"""

from __future__ import annotations

from app.ai.base import Candidate, SceneAnalysis


class NoopVLM:
    name = "noop"

    @property
    def available(self) -> bool:
        return False

    def analyse_scene(self, keyframe_jpeg: bytes, candidates: list[Candidate], *,
                      context: str = "") -> SceneAnalysis | None:
        return None
