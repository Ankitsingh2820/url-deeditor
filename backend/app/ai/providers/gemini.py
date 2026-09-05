"""Gemini vision provider.

Same contract as the Claude provider: one call per scene, output constrained to
RESPONSE_SCHEMA via `response_json_schema`, so both return an identical shape and
can be swapped with a config value.
"""

from __future__ import annotations

import json
import logging

from app.ai.base import RESPONSE_SCHEMA, Candidate, SceneAnalysis
from app.ai.prompts import SYSTEM, scene_user_text

log = logging.getLogger(__name__)


class GeminiVLM:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = None

    @property
    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def analyse_scene(self, keyframe_jpeg: bytes, candidates: list[Candidate], *,
                      context: str = "") -> SceneAnalysis | None:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=keyframe_jpeg, mime_type="image/jpeg"),
            types.Part.from_text(text=scene_user_text([c.id for c in candidates], context)),
        ]
        for candidate in candidates:
            parts.append(types.Part.from_text(text=f"Candidate {candidate.id}:"))
            parts.append(types.Part.from_bytes(data=candidate.jpeg, mime_type="image/jpeg"))

        try:
            response = self._get_client().models.generate_content(
                model=self._model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                    max_output_tokens=2048,
                    http_options=types.HttpOptions(timeout=int(self._timeout * 1000)),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the SDK raises a wide variety
            log.warning("gemini vlm failed: %s: %s", type(exc).__name__, exc)
            return None

        text = getattr(response, "text", None)
        if not text:
            return None
        try:
            return SceneAnalysis.from_payload(json.loads(text))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("gemini vlm returned unparseable JSON: %s", exc)
            return None
