"""Recover the visual style of on-screen text from pixels.

Detection alone cannot rebuild a video: to re-render a caption you need its
colour, its outline and whether it sat on a solid plate. All of that is
recoverable from the crop, because burned-in text is high contrast by design --
that is the whole reason it is legible over footage.

Method: Otsu split of the crop into glyph vs background (the glyph class is the
minority one), then measure each population. Cheap, no model, and it degrades to
sensible defaults on a crop that is not really text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

#: below this contrast the crop is probably not text; report low confidence
MIN_CONTRAST = 25.0
#: background darker/brighter than the ring by this much implies an outline
STROKE_DELTA = 40.0
#: a uniform background behind the glyphs means a solid caption plate
BG_UNIFORM_STD = 24.0

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)


@dataclass
class TextStyle:
    color: str = "#FFFFFF"
    stroke: str | None = None
    bg_color: str | None = None
    has_bg_box: bool = False
    font_px_norm: float = 0.05
    align: str = "center"
    contrast: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def to_hex(bgr: tuple[float, float, float]) -> str:
    b, g, r = (int(max(0, min(255, round(c)))) for c in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(bgr: tuple[float, float, float]) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def _mean_bgr(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        return (0.0, 0.0, 0.0)
    pixels = image[mask.astype(bool)]
    return tuple(float(v) for v in pixels.mean(axis=0))  # type: ignore[return-value]


def extract_style(crop: np.ndarray) -> TextStyle:
    """Estimate colour, outline and plate from a BGR crop of a text box."""
    style = TextStyle()
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return style

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bright = binary.astype(bool)
    # Glyphs cover less area than what they sit on; whichever class is the
    # minority is the text.
    glyph = bright if bright.mean() <= 0.5 else ~bright
    if not glyph.any() or glyph.all():
        return style

    # Antialiasing blends glyph and background over a 1 px rim, which drags a
    # naive mean toward the wrong colour. Measure the eroded *core* for the text
    # and everything beyond the rim for the background; the band in between is
    # where an outline would live.
    glyph_u8 = glyph.astype(np.uint8)
    core = cv2.erode(glyph_u8, _K3, iterations=1).astype(bool)
    if not core.any():
        core = glyph
    near = cv2.dilate(glyph_u8, _K3, iterations=1).astype(bool)
    outer_band = cv2.dilate(glyph_u8, _K5, iterations=1).astype(bool)
    ring = outer_band & ~near
    background = ~outer_band
    if not background.any():
        background = ~glyph

    text_bgr = _mean_bgr(crop, core)
    bg_bgr = _mean_bgr(crop, background)
    contrast = abs(luminance(text_bgr) - luminance(bg_bgr))

    style.color = to_hex(text_bgr)
    style.bg_color = to_hex(bg_bgr)
    style.contrast = round(contrast, 2)
    # Low contrast means the Otsu split found texture, not glyphs.
    style.confidence = round(min(1.0, contrast / 128.0), 3)

    # A uniform background is a caption plate; footage behind text is never flat.
    bg_pixels = gray[background]
    style.has_bg_box = bool(contrast > MIN_CONTRAST and bg_pixels.std() < BG_UNIFORM_STD)

    # An outline shows up as a band hugging the glyphs whose luminance sits
    # apart from the background's.
    if ring.any():
        ring_bgr = _mean_bgr(crop, ring)
        if abs(luminance(ring_bgr) - luminance(bg_bgr)) > STROKE_DELTA:
            style.stroke = to_hex(ring_bgr)

    return style
