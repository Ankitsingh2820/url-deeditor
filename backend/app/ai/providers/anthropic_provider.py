"""Claude vision provider.

Uses the official `anthropic` SDK with `output_config.format` so the reply is a
JSON document matching RESPONSE_SCHEMA -- no prose parsing, no retry-on-garbage
loop. Effort is set low because this is a bounded classification task, not
open-ended reasoning; the pipeline calls it once per scene.
"""

from __future__ import annotations

import base64
import json
import logging

from app.ai.base import RESPONSE_SCHEMA, Candidate, SceneAnalysis
from app.ai.prompts import SYSTEM, scene_user_text

log = logging.getLogger(__name__)


class AnthropicVLM:
    name = "claude"

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
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    @staticmethod
    def _image(jpeg: bytes) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("ascii"),
            },
        }

    def analyse_scene(self, keyframe_jpeg: bytes, candidates: list[Candidate], *,
                      context: str = "") -> SceneAnalysis | None:
        import anthropic

        content: list[dict] = [self._image(keyframe_jpeg)]
        content.append({"type": "text", "text": scene_user_text([c.id for c in candidates],
                                                                context)})
        for candidate in candidates:
            content.append({"type": "text", "text": f"Candidate {candidate.id}:"})
            content.append(self._image(candidate.jpeg))

        try:
            response = self._get_client().messages.create(
                model=self._model,
                max_tokens=2048,
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                # Bounded classification: keep the spend proportional to the task.
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
            )
        except anthropic.APIStatusError as exc:
            log.warning("claude vlm returned %s: %s", exc.status_code, exc.message)
            return None
        except anthropic.APIConnectionError as exc:
            log.warning("claude vlm unreachable: %s", exc)
            return None

        if response.stop_reason == "refusal":
            log.warning("claude vlm refused the request")
            return None

        # output_config.format guarantees the first text block is valid JSON.
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            return None
        try:
            return SceneAnalysis.from_payload(json.loads(text))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("claude vlm returned unparseable JSON: %s", exc)
            return None
