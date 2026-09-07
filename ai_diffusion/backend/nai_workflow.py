"""NovelAI image generation parameter preparation.

Converts plugin UI state into NAI API request bodies.
This module is independent from the ComfyUI workflow system.
"""

from __future__ import annotations

from enum import Enum

from ..image import Extent, Image
from ..settings import ImageFileFormat

# ---------------------------------------------------------------------------
# NAI model identifiers
# ---------------------------------------------------------------------------


class NaiModel(Enum):
    # V5 models (latest)
    v5_curated = "nai-diffusion-5-curated"
    v5_full = "nai-diffusion-5-full"
    # V4.5 models
    v4_5_curated = "nai-diffusion-4-5-curated"
    v4_5_full = "nai-diffusion-4-5-full"
    # V4 models
    v4_curated = "nai-diffusion-4-curated-preview"
    v4_full = "nai-diffusion-4-full"
    # V3 models
    v3 = "nai-diffusion-3"
    v3_inpaint = "nai-diffusion-3-inpainting"
    v3_furry = "nai-diffusion-furry-3"
    v3_furry_inpaint = "nai-diffusion-furry-3-inpainting"

    @staticmethod
    def default():
        return NaiModel.v5_curated

    @staticmethod
    def list_generate():
        """Models available for text-to-image / img2img."""
        return [
            NaiModel.v5_curated,
            NaiModel.v5_full,
            NaiModel.v4_5_curated,
            NaiModel.v4_5_full,
            NaiModel.v4_curated,
            NaiModel.v4_full,
            NaiModel.v3,
        ]

    @staticmethod
    def list_display():
        """Human-readable names for UI display."""
        return {
            NaiModel.v5_curated: "NAI Diffusion V5 (Curated)",
            NaiModel.v5_full: "NAI Diffusion V5 (Full)",
            NaiModel.v4_5_curated: "NAI Diffusion V4.5 (Curated)",
            NaiModel.v4_5_full: "NAI Diffusion V4.5 (Full)",
            NaiModel.v4_curated: "NAI Diffusion V4 (Curated)",
            NaiModel.v4_full: "NAI Diffusion V4 (Full)",
            NaiModel.v3: "NAI Diffusion V3 (Anime)",
        }

    @property
    def display_name(self):
        return NaiModel.list_display().get(self, self.value)

    @property
    def is_v4_5(self):
        return self in (NaiModel.v4_5_curated, NaiModel.v4_5_full)

    @property
    def is_v5(self):
        return self in (NaiModel.v5_curated, NaiModel.v5_full)

    @property
    def is_v4(self):
        """Whether this model uses the V4+ structured prompt request fields."""
        return self in (
            NaiModel.v5_curated,
            NaiModel.v5_full,
            NaiModel.v4_5_curated,
            NaiModel.v4_5_full,
            NaiModel.v4_curated,
            NaiModel.v4_full,
        )

    @property
    def is_v3(self):
        return self in (
            NaiModel.v3,
            NaiModel.v3_inpaint,
            NaiModel.v3_furry,
            NaiModel.v3_furry_inpaint,
        )

    @property
    def is_curated(self):
        return self in (NaiModel.v5_curated, NaiModel.v4_5_curated, NaiModel.v4_curated)

    @property
    def supports_vibe(self):
        return not self.is_v5

    @property
    def supports_precise_reference(self):
        return self.is_v4_5

    @property
    def supports_variety_plus(self):
        return not self.is_v5

    @property
    def inpaint_model(self):
        """Return the corresponding legacy inpaint enum when one exists."""
        if self is NaiModel.v3:
            return NaiModel.v3_inpaint
        if self is NaiModel.v3_furry:
            return NaiModel.v3_furry_inpaint
        return self

    @property
    def inpaint_model_name(self):
        if self is NaiModel.v5_curated:
            # The official V5 launch temporarily routes Curated inpainting through
            # V4.5 Curated until the dedicated V5 Curated inpainting model ships.
            return f"{NaiModel.v4_5_curated.value}-inpainting"
        if self.is_v3:
            return self.inpaint_model.value
        return f"{self.value}-inpainting"


# ---------------------------------------------------------------------------
# NAI sampler names
# ---------------------------------------------------------------------------


class NaiSampler(Enum):
    k_euler = "k_euler"
    k_euler_ancestral = "k_euler_ancestral"
    k_dpmpp_2m = "k_dpmpp_2m"
    k_dpmpp_2m_sde = "k_dpmpp_2m_sde"
    k_dpmpp_sde = "k_dpmpp_sde"
    ddim_v3 = "ddim_v3"

    @staticmethod
    def default():
        return NaiSampler.k_euler_ancestral


# Map from ComfyUI sampler names to NAI sampler names
_SAMPLER_MAP: dict[str, NaiSampler] = {
    "euler": NaiSampler.k_euler,
    "euler_ancestral": NaiSampler.k_euler_ancestral,
    "dpmpp_2m": NaiSampler.k_dpmpp_2m,
    "dpmpp_2m_sde": NaiSampler.k_dpmpp_2m_sde,
    "dpmpp_2m_sde_gpu": NaiSampler.k_dpmpp_2m_sde,
    "dpmpp_sde": NaiSampler.k_dpmpp_sde,
    "dpmpp_sde_gpu": NaiSampler.k_dpmpp_sde,
    "ddim": NaiSampler.ddim_v3,
}


def map_sampler(comfy_sampler: str) -> NaiSampler:
    """Map a ComfyUI sampler name to a NAI sampler."""
    return _SAMPLER_MAP.get(comfy_sampler, NaiSampler.default())


# ---------------------------------------------------------------------------
# NAI noise schedules
# ---------------------------------------------------------------------------


class NaiNoiseSchedule(Enum):
    native = "native"
    karras = "karras"
    exponential = "exponential"
    polyexponential = "polyexponential"

    @staticmethod
    def default():
        return NaiNoiseSchedule.karras


# Map from ComfyUI scheduler names to NAI noise schedule
_SCHEDULE_MAP: dict[str, NaiNoiseSchedule] = {
    "normal": NaiNoiseSchedule.native,
    "karras": NaiNoiseSchedule.karras,
    "exponential": NaiNoiseSchedule.exponential,
    "sgm_uniform": NaiNoiseSchedule.native,
    "ddim_uniform": NaiNoiseSchedule.native,
}


def map_noise_schedule(comfy_scheduler: str) -> NaiNoiseSchedule:
    return _SCHEDULE_MAP.get(comfy_scheduler, NaiNoiseSchedule.default())


# ---------------------------------------------------------------------------
# NAI UC (Undesired Content) presets
# ---------------------------------------------------------------------------


class NaiUCPreset(Enum):
    heavy = 0
    light = 1
    none = 2

    @property
    def display_name(self):
        return {
            NaiUCPreset.heavy: "Heavy",
            NaiUCPreset.light: "Light",
            NaiUCPreset.none: "None",
        }[self]


# ---------------------------------------------------------------------------
# NAI action types
# ---------------------------------------------------------------------------


class NaiAction(Enum):
    generate = "generate"
    img2img = "img2img"
    infill = "infill"


# ---------------------------------------------------------------------------
# Resolution utilities
# ---------------------------------------------------------------------------

# NAI requires width/height to be multiples of 64.
NAI_RESOLUTION_MULTIPLE = 64
NAI_MIN_SIDE = 64
NAI_MAX_SIDE = 2048
NAI_FREE_PIXELS = 1024 * 1024

# Maximum total pixel count NAI accepts for a generation request.
# 3145728 is the official cap (novelai.net web build, cross-checked against the
# launcher's NaiResolutionAdapter.officialMaxPixels).
# NOTE: 1024*1024 = 1048576 is NOT the API limit — it is only the threshold below
# which Opus subscribers generate for free. Clamping to it silently shrank valid
# canvases (e.g. 1600x896 came back as 1344x768).
NAI_MAX_PIXELS = 3_145_728
NAI_EDIT_MAX_PIXELS = NAI_MAX_PIXELS


def nai_resolution_value(value: int) -> int:
    value = max(NAI_MIN_SIDE, min(int(value), NAI_MAX_SIDE))
    return max(
        NAI_MIN_SIDE,
        min(
            ((value + NAI_RESOLUTION_MULTIPLE // 2) // NAI_RESOLUTION_MULTIPLE)
            * NAI_RESOLUTION_MULTIPLE,
            NAI_MAX_SIDE,
        ),
    )


def nai_resolution(extent: Extent) -> Extent:
    """Snap both sides to the nearest valid NAI value."""
    return Extent(nai_resolution_value(extent.width), nai_resolution_value(extent.height))


def _reduce_to_pixel_limit(extent: Extent, source: Extent, max_pixels: int) -> Extent:
    result = extent
    source_aspect = source.width / max(source.height, 1)
    while result.pixel_count > max_pixels:
        candidates = []
        if result.width > NAI_MIN_SIDE:
            candidates.append(Extent(result.width - NAI_RESOLUTION_MULTIPLE, result.height))
        if result.height > NAI_MIN_SIDE:
            candidates.append(Extent(result.width, result.height - NAI_RESOLUTION_MULTIPLE))
        if result.width > NAI_MIN_SIDE and result.height > NAI_MIN_SIDE:
            # Shrinking both sides at once is the only move that preserves the
            # ratio. Without it the loop can step sideways only: a square source
            # snapped up to 1792x1792 came back as 1728x1792, because the legal
            # and exactly square 1728x1728 was never a candidate.
            candidates.append(
                Extent(
                    result.width - NAI_RESOLUTION_MULTIPLE,
                    result.height - NAI_RESOLUTION_MULTIPLE,
                )
            )
        if not candidates:
            break
        result = min(
            candidates,
            key=lambda candidate: (
                abs(candidate.width / candidate.height - source_aspect),
                -candidate.pixel_count,
            ),
        )
    return result


def nai_free_resolution(extent: Extent) -> Extent:
    """Largest free-tier resolution with the same aspect ratio.

    Free means total pixel count <= NAI_FREE_PIXELS (with <= 28 steps), not a
    per-side limit — 1920x512 is just as free as 1024x1024. Scales up as well as
    down, so the result is always the largest legal size for the given ratio.
    """
    source = Extent(max(extent.width, 1), max(extent.height, 1))
    target = source.scale_to_pixel_count(NAI_FREE_PIXELS)
    return _reduce_to_pixel_limit(nai_resolution(target), source, NAI_FREE_PIXELS)


def nai_auto_resolution(extent: Extent) -> Extent:
    """Choose an edit resolution from the source size while preserving its ratio."""
    source = Extent(max(extent.width, 1), max(extent.height, 1))
    scale = 1.0
    pixel_limit = NAI_EDIT_MAX_PIXELS
    if source.pixel_count < NAI_FREE_PIXELS:
        scale = (NAI_FREE_PIXELS / source.pixel_count) ** 0.5
        pixel_limit = NAI_FREE_PIXELS
    elif source.pixel_count > NAI_EDIT_MAX_PIXELS:
        scale = (NAI_EDIT_MAX_PIXELS / source.pixel_count) ** 0.5

    scale = min(scale, NAI_MAX_SIDE / source.longest_side)
    target = nai_resolution(Extent(round(source.width * scale), round(source.height * scale)))
    return _reduce_to_pixel_limit(target, source, pixel_limit)


def clamp_resolution(extent: Extent, max_pixels: int = NAI_MAX_PIXELS) -> Extent:
    """Scale down if total pixel count exceeds NAI limits, then align to 64."""
    total = extent.width * extent.height
    if total <= max_pixels:
        target = nai_resolution(extent)
        return _reduce_to_pixel_limit(target, extent, max_pixels)

    scale = (max_pixels / total) ** 0.5
    w = int(extent.width * scale)
    h = int(extent.height * scale)
    target = nai_resolution(Extent(w, h))
    return _reduce_to_pixel_limit(target, extent, max_pixels)


def nai_edit_resolution(extent: Extent) -> Extent:
    return clamp_resolution(extent, NAI_EDIT_MAX_PIXELS)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def image_to_base64(image: Image) -> str:
    """Encode an Image to base64 PNG string for NAI API."""
    return image.to_base64(ImageFileFormat.png)


# ---------------------------------------------------------------------------
# Precise (director) reference image preparation
# ---------------------------------------------------------------------------


def prepare_nai_precise_reference_image(image: Image) -> Image:
    """Normalize a precise-reference image the way the NAI web client (Q6) does.

    Ported from the launcher's NAIApiUtils.ensurePngFormat: pick the standard
    canvas — (1024,1536) portrait / (1536,1024) landscape / (1472,1472) square —
    whose aspect ratio is closest to the source, aspect-preserving downscale,
    then center-paste onto a black RGB background.
    """
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtGui import QColor as _QColor
    from PyQt5.QtGui import QImage as _QImage
    from PyQt5.QtGui import QPainter as _QPainter

    src = image._qimage
    w, h = max(src.width(), 1), max(src.height(), 1)
    aspect = w / h
    tw, th = min(
        [(1024, 1536), (1536, 1024), (1472, 1472)],
        key=lambda c: abs(aspect - c[0] / c[1]),
    )
    scale = min(tw / w, th / h)
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    scaled = src.scaled(
        nw,
        nh,
        _Qt.AspectRatioMode.IgnoreAspectRatio,
        _Qt.TransformationMode.SmoothTransformation,
    )
    canvas = _QImage(tw, th, _QImage.Format.Format_RGB32)
    canvas.fill(_QColor(0, 0, 0))
    painter = _QPainter(canvas)
    painter.drawImage((tw - nw) // 2, (th - nh) // 2, scaled)
    painter.end()
    return Image(canvas)
