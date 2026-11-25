"""NovelAI image generation parameter preparation.

Converts plugin UI state into NAI API request bodies.
This module is independent from the ComfyUI workflow system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..image import Bounds, Extent, Image
from ..settings import ImageFileFormat, NaiAutoSize

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
        return self.is_v4_5 or self.is_v5

    @property
    def supports_outfit_reference(self):
        return self.is_v5

    @property
    def supports_precise_reference_fidelity(self):
        return self.is_v4_5

    @property
    def supports_precise_reference_inpainting(self):
        return self.is_v4_5

    @property
    def bucket_precise_reference(self):
        return self.is_v5

    @property
    def precise_reference_strength_max(self):
        return 5.0 if self.is_v5 else 1.0

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


def nai_auto_resolution(extent: Extent, size: NaiAutoSize = NaiAutoSize.best_match) -> Extent:
    """Choose an edit resolution from the source size while preserving its ratio.

    `size` is the user's policy: `small` keeps the request inside NAI's 1 MP tier
    (the free-tier cap, same as the docker's 小 button), `best_match` picks the
    largest legal size for that ratio. Below 1 MP both agree, since NAI output
    smaller than that is not worth generating.
    """
    if size is NaiAutoSize.small:
        return nai_free_resolution(extent)
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


_V5_PRECISE_REFERENCE_BUCKETS = (
    (256, 4096),
    (256, 4032),
    (256, 3968),
    (256, 3904),
    (256, 3840),
    (256, 3776),
    (256, 3712),
    (256, 3648),
    (256, 3584),
    (256, 3520),
    (256, 3456),
    (256, 3392),
    (256, 3328),
    (320, 3264),
    (320, 3200),
    (320, 3136),
    (320, 3072),
    (320, 3008),
    (320, 2944),
    (320, 2880),
    (320, 2816),
    (320, 2752),
    (384, 2688),
    (384, 2624),
    (384, 2560),
    (384, 2496),
    (384, 2432),
    (384, 2368),
    (448, 2304),
    (448, 2240),
    (448, 2176),
    (448, 2112),
    (512, 2048),
    (512, 1984),
    (512, 1920),
    (512, 1856),
    (576, 1792),
    (576, 1728),
    (576, 1664),
    (640, 1600),
    (640, 1536),
    (704, 1472),
    (704, 1408),
    (768, 1344),
    (768, 1280),
    (832, 1216),
    (896, 1152),
    (960, 1088),
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1344, 768),
    (1408, 704),
    (1472, 704),
    (1536, 640),
    (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
    (2112, 448),
    (2176, 448),
    (2240, 448),
    (2304, 448),
    (2368, 384),
    (2432, 384),
    (2496, 384),
    (2560, 384),
    (2624, 384),
    (2688, 384),
    (2752, 320),
    (2816, 320),
    (2880, 320),
    (2944, 320),
    (3008, 320),
    (3072, 320),
    (3136, 320),
    (3200, 320),
    (3264, 320),
    (3328, 256),
    (3392, 256),
    (3456, 256),
    (3520, 256),
    (3584, 256),
    (3648, 256),
    (3712, 256),
    (3776, 256),
    (3840, 256),
    (3904, 256),
    (3968, 256),
    (4032, 256),
    (4096, 256),
)


def bucket_nai_precise_reference_image(image: Image) -> Image:
    """Scale and center-crop a precise-reference image into NAI V5's 97-bucket canvas."""
    from PyQt5.QtCore import Qt as _Qt

    src = image._qimage
    w, h = max(src.width(), 1), max(src.height(), 1)
    log_aspect = math.log(w / h)
    bw, bh = min(
        _V5_PRECISE_REFERENCE_BUCKETS,
        key=lambda c: abs(log_aspect - math.log(c[0] / c[1])),
    )
    if src.width() == bw and src.height() == bh:
        return image
    scale = max(bw / w, bh / h)
    rw = max(math.floor(w * scale + 0.5), bw)
    rh = max(math.floor(h * scale + 0.5), bh)
    left = (rw - bw) // 2
    top = (rh - bh) // 2
    scaled = src.scaled(
        rw,
        rh,
        _Qt.AspectRatioMode.IgnoreAspectRatio,
        _Qt.TransformationMode.SmoothTransformation,
    )
    return Image(scaled.copy(left, top, bw, bh))


def prepare_nai_precise_reference_image(image: Image, *, bucket: bool = False) -> Image:
    """Normalize a precise-reference image for V4.5 (3 black-padded canvases) or V5 (97 buckets)."""
    if bucket:
        return bucket_nai_precise_reference_image(image)
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


# ---------------------------------------------------------------------------
# Auto crop box & tiled img2img layout/blending
# ---------------------------------------------------------------------------


def compute_auto_crop_bounds(canvas_extent: Extent, mask_bounds: Bounds) -> Bounds:
    """Compute a centered crop box up to NAI_FREE_PIXELS matching the mask's aspect ratio."""
    cw_max, ch_max = canvas_extent.width, canvas_extent.height
    mw = max(mask_bounds.width, 1)
    mh = max(mask_bounds.height, 1)

    # Clamp aspect ratio to [1/3, 3] so thin strokes still get useful context
    aspect = max(1.0 / 3.0, min(3.0, mw / mh))
    ideal_h = math.sqrt(NAI_FREE_PIXELS / aspect)
    ideal_w = ideal_h * aspect
    free = nai_free_resolution(
        Extent(max(NAI_MIN_SIDE, round(ideal_w)), max(NAI_MIN_SIDE, round(ideal_h)))
    )

    box_w = min(max(free.width, mw), cw_max)
    box_h = min(max(free.height, mh), ch_max)

    cx = mask_bounds.x + mask_bounds.width / 2.0
    cy = mask_bounds.y + mask_bounds.height / 2.0
    x = max(0, min(round(cx - box_w / 2.0), cw_max - box_w))
    y = max(0, min(round(cy - box_h / 2.0), ch_max - box_h))
    return Bounds(x, y, box_w, box_h)


@dataclass(frozen=True)
class NaiTileSpec:
    bounds: Bounds
    target_extent: Extent
    overlap_left: int = 0
    overlap_top: int = 0


def compute_nai_tile_layout(
    extent: Extent, overlap_ratio: float = 0.10, max_pixels: int = NAI_FREE_PIXELS
) -> list[NaiTileSpec]:
    """Divide a canvas into minimal, evenly-sized tiles within free pixel limits with overlap."""
    w, h = max(extent.width, 1), max(extent.height, 1)
    if w * h <= max_pixels and max(w, h) <= NAI_MAX_SIDE:
        return [NaiTileSpec(Bounds(0, 0, w, h), nai_free_resolution(extent))]

    def _tile_dim(total: int, count: int) -> int:
        if count <= 1:
            return total
        return min(total, math.ceil(total / (1.0 + (count - 1) * (1.0 - overlap_ratio))))

    best_cols, best_rows = 1, 1
    best_score: tuple[int, float] | None = None

    max_cols = min(max(math.ceil(w / NAI_MIN_SIDE), 1), 32)
    max_rows = min(max(math.ceil(h / NAI_MIN_SIDE), 1), 32)

    for cols in range(1, max_cols + 1):
        tw = _tile_dim(w, cols)
        if tw > NAI_MAX_SIDE:
            continue
        for rows in range(1, max_rows + 1):
            th = _tile_dim(h, rows)
            if th > NAI_MAX_SIDE or tw * th > max_pixels:
                continue
            count = cols * rows
            aspect_penalty = max(tw / max(th, 1), th / max(tw, 1))
            score = (count, aspect_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_cols, best_rows = cols, rows

    # Fallback if canvas is huge (> 32x32 tiles)
    if best_score is None:
        side = int(math.sqrt(max_pixels))
        best_cols = max(1, math.ceil(w / max(side * (1.0 - overlap_ratio), 1)))
        best_rows = max(1, math.ceil(h / max(side * (1.0 - overlap_ratio), 1)))

    tile_w = _tile_dim(w, best_cols)
    tile_h = _tile_dim(h, best_rows)

    def _positions(total: int, tile_size: int, count: int) -> list[int]:
        if count <= 1:
            return [0]
        step = (total - tile_size) / (count - 1)
        pos = [round(i * step) for i in range(count)]
        pos[-1] = total - tile_size
        return pos

    xs = _positions(w, tile_w, best_cols)
    ys = _positions(h, tile_h, best_rows)
    target = nai_free_resolution(Extent(tile_w, tile_h))

    tiles: list[NaiTileSpec] = []
    for r, y in enumerate(ys):
        overlap_top = max(0, (ys[r - 1] + tile_h) - y) if r > 0 else 0
        for c, x in enumerate(xs):
            overlap_left = max(0, (xs[c - 1] + tile_w) - x) if c > 0 else 0
            tiles.append(
                NaiTileSpec(
                    Bounds(x, y, tile_w, tile_h),
                    target,
                    overlap_left=overlap_left,
                    overlap_top=overlap_top,
                )
            )
    return tiles


def feather_tile_edges(image: Image, overlap_left: int, overlap_top: int) -> Image:
    """Apply linear alpha fade-in on left/top overlap strips so sequential tiles blend seamlessly."""
    if overlap_left <= 0 and overlap_top <= 0:
        return image
    from PyQt5.QtGui import QColor as _QColor
    from PyQt5.QtGui import QImage as _QImage
    from PyQt5.QtGui import QLinearGradient as _QLinearGradient
    from PyQt5.QtGui import QPainter as _QPainter

    patch = image._qimage.convertToFormat(_QImage.Format.Format_ARGB32)
    w, h = patch.width(), patch.height()
    if overlap_left > 0:
        mask = _QImage(w, h, _QImage.Format.Format_ARGB32)
        mask.fill(_QColor(0, 0, 0, 255))
        grad = _QLinearGradient(0, 0, min(overlap_left, w), 0)
        grad.setColorAt(0.0, _QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, _QColor(0, 0, 0, 255))
        mp = _QPainter(mask)
        mp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_Source)
        mp.fillRect(0, 0, min(overlap_left, w), h, grad)
        mp.end()
        pp = _QPainter(patch)
        pp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_DestinationIn)
        pp.drawImage(0, 0, mask)
        pp.end()
    if overlap_top > 0:
        mask = _QImage(w, h, _QImage.Format.Format_ARGB32)
        mask.fill(_QColor(0, 0, 0, 255))
        grad = _QLinearGradient(0, 0, 0, min(overlap_top, h))
        grad.setColorAt(0.0, _QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, _QColor(0, 0, 0, 255))
        mp = _QPainter(mask)
        mp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_Source)
        mp.fillRect(0, 0, w, min(overlap_top, h), grad)
        mp.end()
        pp = _QPainter(patch)
        pp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_DestinationIn)
        pp.drawImage(0, 0, mask)
        pp.end()
    return Image(patch)


def blend_nai_tiles(extent: Extent, tiles: list[NaiTileSpec], images: list[Image]) -> Image:
    """Seamlessly blend tiled img2img results across their overlapping strips."""
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtGui import QColor as _QColor
    from PyQt5.QtGui import QImage as _QImage
    from PyQt5.QtGui import QLinearGradient as _QLinearGradient
    from PyQt5.QtGui import QPainter as _QPainter

    if len(tiles) == 1:
        img = images[0]
        if img.extent != extent:
            img = Image.scale(img, extent)
        return img

    # Group tiles by row (y coordinate)
    rows: dict[int, list[tuple[NaiTileSpec, Image]]] = {}
    for tile, img in zip(tiles, images):
        scaled = img if img.extent == tile.bounds.extent else Image.scale(img, tile.bounds.extent)
        rows.setdefault(tile.bounds.y, []).append((tile, scaled))

    sorted_ys = sorted(rows.keys())
    row_strips: list[tuple[int, int, _QImage]] = []

    def _feather_patch(qimg: _QImage, overlap: int, horizontal: bool) -> _QImage:
        patch = qimg.convertToFormat(_QImage.Format.Format_ARGB32)
        if overlap <= 0:
            return patch
        mask = _QImage(patch.width(), patch.height(), _QImage.Format.Format_ARGB32)
        mask.fill(_QColor(0, 0, 0, 255))
        grad = (
            _QLinearGradient(0, 0, overlap, 0) if horizontal else _QLinearGradient(0, 0, 0, overlap)
        )
        grad.setColorAt(0.0, _QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, _QColor(0, 0, 0, 255))
        mp = _QPainter(mask)
        mp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_Source)
        if horizontal:
            mp.fillRect(0, 0, overlap, patch.height(), grad)
        else:
            mp.fillRect(0, 0, patch.width(), overlap, grad)
        mp.end()

        pp = _QPainter(patch)
        pp.setCompositionMode(_QPainter.CompositionMode.CompositionMode_DestinationIn)
        pp.drawImage(0, 0, mask)
        pp.end()
        return patch

    for y in sorted_ys:
        row_items = sorted(rows[y], key=lambda item: item[0].bounds.x)
        row_h = row_items[0][0].bounds.height
        strip = _QImage(extent.width, row_h, _QImage.Format.Format_ARGB32)
        strip.fill(_Qt.GlobalColor.transparent)
        sp = _QPainter(strip)
        prev_right = 0
        for idx, (tile, img) in enumerate(row_items):
            if idx == 0:
                sp.drawImage(tile.bounds.x, 0, img._qimage)
            else:
                overlap = max(0, prev_right - tile.bounds.x)
                patch = _feather_patch(img._qimage, overlap, horizontal=True)
                sp.drawImage(tile.bounds.x, 0, patch)
            prev_right = tile.bounds.x + tile.bounds.width
        sp.end()
        row_strips.append((y, row_h, strip))

    canvas = _QImage(extent.width, extent.height, _QImage.Format.Format_ARGB32)
    canvas.fill(_Qt.GlobalColor.transparent)
    cp = _QPainter(canvas)
    prev_bottom = 0
    for idx, (y, row_h, strip) in enumerate(row_strips):
        if idx == 0:
            cp.drawImage(0, y, strip)
        else:
            overlap = max(0, prev_bottom - y)
            patch = _feather_patch(strip, overlap, horizontal=False)
            cp.drawImage(0, y, patch)
        prev_bottom = y + row_h
    cp.end()
    return Image(canvas)
