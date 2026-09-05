"""Prompt text for the scene analysis call.

Kept apart from the providers so the wording can be tuned without touching
transport code, and so both providers are demonstrably asking the same thing.
"""

from __future__ import annotations

SYSTEM = """You analyse frames from short-form social video (TikTok, Reels, Meta \
ads) for a de-editing tool that takes an edited video apart into its components.

You are given one keyframe from a scene, then a series of cropped regions that a \
computer-vision pass has flagged as *possible* composited graphics. Your job is to \
describe the shot and rule on each crop.

A region is an overlay only if it was added in editing: a product pop-up, a \
sticker, a logo bug, a UI element, a pasted screenshot. It is NOT an overlay if it \
is part of the filmed world -- a poster on the wall, a phone the person is holding, \
a label on a real bottle, a picture frame, a window. This distinction is the whole \
point of the task, so be strict: when a region is plausibly physical, say \
is_overlay false.

Be concise. The description is one sentence. Reply for every candidate id you are \
given, and only those ids."""


def scene_user_text(candidate_ids: list[str], context: str = "") -> str:
    """The text that accompanies the keyframe and crops."""
    lines = ["This is the keyframe for one scene."]
    if context:
        lines.append(context)
    if candidate_ids:
        lines.append(
            f"\n{len(candidate_ids)} candidate region(s) follow, in this order: "
            + ", ".join(candidate_ids)
            + ". Rule on each one."
        )
    else:
        lines.append("\nThere are no candidate regions; return an empty overlays list.")
    return "\n".join(lines)
