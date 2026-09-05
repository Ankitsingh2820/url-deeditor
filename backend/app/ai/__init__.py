"""Vision-language integration, behind one swappable interface."""

from __future__ import annotations

import logging

from app.ai.base import Candidate, OverlayVerdict, SceneAnalysis, VLMProvider
from app.config import settings

log = logging.getLogger(__name__)


def get_provider(name: str | None = None) -> VLMProvider:
    """Build the configured provider, or a no-op when it cannot be used.

    Never raises: an unconfigured or uninstalled provider degrades to the no-op
    so a missing API key can never fail a job.
    """
    from app.ai.providers.noop import NoopVLM

    if not settings.ai_enabled:
        return NoopVLM()

    choice = (name or settings.vlm_provider).lower()
    provider: VLMProvider
    if choice == "gemini":
        from app.ai.providers.gemini import GeminiVLM

        provider = GeminiVLM(settings.gemini_api_key, settings.gemini_model,
                             settings.vlm_timeout_s)
    elif choice in ("claude", "anthropic"):
        from app.ai.providers.anthropic_provider import AnthropicVLM

        provider = AnthropicVLM(settings.anthropic_api_key, settings.anthropic_model,
                                settings.vlm_timeout_s)
    else:
        return NoopVLM()

    if not provider.available:
        log.info("VLM provider %r has no credentials; continuing without it", choice)
        return NoopVLM()
    return provider


__all__ = ["Candidate", "OverlayVerdict", "SceneAnalysis", "VLMProvider", "get_provider"]
