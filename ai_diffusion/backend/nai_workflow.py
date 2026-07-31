"""NovelAI image generation parameter preparation.

Converts plugin UI state into NAI API request bodies.
This module is independent from the ComfyUI workflow system.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from ..image import Extent, Image, multiple_of
from ..settings import ImageFileFormat
from ..util import client_logger as log


# ---------------------------------------------------------------------------
# NAI model identifiers
# ---------------------------------------------------------------------------

class NaiModel(Enum):
    # V4.5 models (latest, default)
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
        return NaiModel.v4_5_curated

    @staticmethod
    def list_generate():
        """Models available for text-to-image / img2img."""
        return [
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
    def is_v4(self):
        return self in (
            NaiModel.v4_5_curated, NaiModel.v4_5_full,
            NaiModel.v4_curated, NaiModel.v4_full,
        )

    @property
    def is_v3(self):
        return self in (NaiModel.v3, NaiModel.v3_inpaint, NaiModel.v3_furry, NaiModel.v3_furry_inpaint)

    @property
    def is_curated(self):
        return self in (NaiModel.v4_5_curated, NaiModel.v4_curated)

    @property
    def inpaint_model(self):
        """Return the corresponding inpaint model, or self if V4/V4.5 (uses same model)."""
        if self is NaiModel.v3:
            return NaiModel.v3_inpaint
        if self is NaiModel.v3_furry:
            return NaiModel.v3_furry_inpaint
        return self


# ---------------------------------------------------------------------------
# Quality tags
# ---------------------------------------------------------------------------

# Model-specific quality tags appended to the prompt when "Add Quality Tags" is
# on. NAI's own client appends these strings client-side, and the website derives
# the "Add Quality Tags" indicator from their PRESENCE in the prompt — not from
# the qualityToggle request field alone. So we must append them ourselves for the
# generated image to report the toggle as on (and to actually get the boost).
# Ported verbatim from the reference launcher's QualityTags (api_constants.dart).
_NAI_QUALITY_TAGS: dict[NaiModel, str] = {
    NaiModel.v4_5_full: "location, very aesthetic, masterpiece, no text",
    NaiModel.v4_5_curated: "location, masterpiece, no text, -0.8::feet::, rating:general",
    NaiModel.v4_full: "no text, best quality, very aesthetic, absurdres",
    NaiModel.v4_curated: "rating:general, amazing quality, very aesthetic, absurdres",
    NaiModel.v3: "best quality, amazing quality, very aesthetic, absurdres",
    NaiModel.v3_furry: "{best quality}, {amazing quality}",
}


def apply_quality_tags(prompt: str, model: NaiModel) -> str:
    """Append the model's quality tags to the prompt (V3+ → post-positioned).

    Mirrors QualityTags.applyQualityTags in the reference launcher: no-op when the
    model has no tags; returns the tags alone for an empty prompt; otherwise joins
    with ", " (or " " if the prompt already ends with a comma).
    """
    tags = _NAI_QUALITY_TAGS.get(model)
    if not tags:
        return prompt
    trimmed = prompt.strip()
    if not trimmed:
        return tags
    if trimmed.endswith(","):
        return f"{trimmed} {tags}"
    return f"{trimmed}, {tags}"


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

# NAI requires width/height to be multiples of 64
_NAI_RESOLUTION_MULTIPLE = 64

# Maximum total pixel count NAI accepts for a generation request.
# 3145728 is the official cap (novelai.net web build, cross-checked against the
# launcher's NaiResolutionAdapter.officialMaxPixels).
# NOTE: 1024*1024 = 1048576 is NOT the API limit — it is only the threshold below
# which Opus subscribers generate for free. Clamping to it silently shrank valid
# canvases (e.g. 1600x896 came back as 1344x768).
_NAI_MAX_PIXELS = 3145728


def nai_resolution(extent: Extent) -> Extent:
    """Adjust extent to be compatible with NAI requirements (multiple of 64)."""
    w = multiple_of(extent.width, _NAI_RESOLUTION_MULTIPLE)
    h = multiple_of(extent.height, _NAI_RESOLUTION_MULTIPLE)
    return Extent(max(w, _NAI_RESOLUTION_MULTIPLE), max(h, _NAI_RESOLUTION_MULTIPLE))


def clamp_resolution(extent: Extent, max_pixels: int = _NAI_MAX_PIXELS) -> Extent:
    """Scale down if total pixel count exceeds NAI limits, then align to 64."""
    total = extent.width * extent.height
    if total <= max_pixels:
        return nai_resolution(extent)

    scale = (max_pixels / total) ** 0.5
    w = int(extent.width * scale)
    h = int(extent.height * scale)
    return nai_resolution(Extent(w, h))


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def image_to_base64(image: Image) -> str:
    """Encode an Image to base64 PNG string for NAI API."""
    return image.to_base64(ImageFileFormat.png)


# ---------------------------------------------------------------------------
# Request body builders
# ---------------------------------------------------------------------------

def build_generate_request(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    model: NaiModel = NaiModel.v4_curated,
    action: NaiAction = NaiAction.generate,
    # Sampling parameters
    sampler: NaiSampler = NaiSampler.k_euler,
    steps: int = 28,
    scale: float = 5.0,
    cfg_rescale: float = 0.0,
    noise_schedule: NaiNoiseSchedule = NaiNoiseSchedule.native,
    seed: int = 0,
    n_samples: int = 1,
    # Quality & UC
    quality_toggle: bool = True,
    uc_preset: NaiUCPreset = NaiUCPreset.heavy,
    # Dynamic thresholding
    dynamic_thresholding: bool = False,
    # Variety+ — when True, skip_cfg_above_sigma is computed from resolution
    # (matches the reference launcher's formula). See buildBaseParameters L97-99.
    variety_plus: bool = False,
    # img2img / inpaint
    image: str | None = None,
    strength: float = 0.7,
    noise: float = 0.0,
    mask: str | None = None,
    # infill: the launcher's "重绘强度" masked-region redraw control. Driven together
    # with `strength` above from Krita's single 重绘幅度 slider (see convert_workflow).
    inpaint_img2img_strength: float | None = None,
    # Vibe Transfer
    reference_image_multiple: list[str] | None = None,
    reference_strength_multiple: list[float] | None = None,
    reference_information_extracted_multiple: list[float] | None = None,
    # Precise (director) reference, V4.5 only. Each item:
    # {"image": b64, "caption": "character"|"style"|"character&style",
    #  "strength": float, "secondary": float}  (secondary = 1 - fidelity)
    precise_references: list[dict[str, Any]] | None = None,
    # V4 structured prompt (optional)
    v4_prompt: dict | None = None,
    v4_negative_prompt: dict | None = None,
) -> dict[str, Any]:
    """Build the full NAI /ai/generate-image request body."""

    is_infill = action is NaiAction.infill

    # noise_schedule: V4/V4.5 do NOT support "native" — the launcher forces it to
    # "karras". Sending "native" to V4 is a primary cause of grid/overexposure.
    # Reference: nai_image_request_builder.dart L83-85.
    ns = noise_schedule.value
    if model.is_v4 and ns == "native":
        ns = "karras"

    # add_original_image: True for ALL actions, including infill. This matches the
    # working reference implementation (ComfyUI_RS_NAI_API_Request, NAIInpaintNode:
    # add_original_image=True + inpaintImg2ImgStrength, no `strength` key). The
    # launcher sends False here and its masked-region strength is NOT honored by
    # the server (pixel-proven 2026-07-29) — so False likely routes the server to
    # the legacy full-repaint infill pipeline. We still composite the patch
    # client-side; the server overlaying originals outside the mask is harmless.
    add_original_image = True

    parameters: dict[str, Any] = {
        "params_version": 3,
        "width": width,
        "height": height,
        "scale": scale,
        "sampler": sampler.value,
        "steps": steps,
        "n_samples": n_samples,
        "ucPreset": uc_preset.value,
        "qualityToggle": quality_toggle,
        "autoSmea": False,
        # dynamic_thresholding (decrisp) only meaningful on V3
        "dynamic_thresholding": dynamic_thresholding and model.is_v3,
        "controlnet_strength": 1,
        "legacy": False,
        "add_original_image": add_original_image,
        "cfg_rescale": cfg_rescale,
        "noise_schedule": ns,
        "normalize_reference_strength_multiple": True,
        "seed": seed,
        # extra_noise_seed seeds the noise that img2img/inpaint adds to the source
        # latents. The NAI web UI sets it to seed - 1 when unset (bundle chunk 2075:
        # `void 0===i.extra_noise_seed&&(i.extra_noise_seed=i.seed-1)`).
        "extra_noise_seed": max(seed - 1, 0),
        "negative_prompt": negative_prompt,
        # These two flags govern Euler Ancestral behaviour; the launcher always
        # sends them. Missing them is a cause of divergent/overfit results.
        "deliberate_euler_ancestral_bug": False,
        "prefer_brownian": True,
    }

    # Variety+ — skip CFG above a resolution-dependent sigma threshold. The
    # launcher always sends this key (null when disabled). Reference L97-99.
    if variety_plus:
        parameters["skip_cfg_above_sigma"] = (
            58.0 * math.sqrt(4.0 * (width / 8) * (height / 8) / 63232)
        )
    else:
        parameters["skip_cfg_above_sigma"] = None

    # SMEA (sm/sm_dyn) and the separate `uc` field are V3-only. V4/V4.5 must NOT
    # receive them. Reference L101-116.
    if not model.is_v4:
        resolution = width * height
        auto_smea = resolution > 1024 * 1024
        is_ddim = "ddim" in sampler.value
        parameters["sm"] = False if is_ddim else auto_smea
        parameters["sm_dyn"] = False
        parameters["uc"] = negative_prompt

    # Masked-region redraw strength (NAI web "Inpainting Strength", V4+ only).
    # THE SERVER-HONORED FIELD IS THE NESTED ``img2img`` OBJECT, not the flat
    # inpaintImg2ImgStrength. Straight from novelai.net's own bundle
    # (chunk 2075, 2026-07):
    #   O.img2imgInpainting && i.mask && (i.inpaintImg2ImgStrength ?? 1) !== 1
    #     ? i.img2img = {strength: i.inpaintImg2ImgStrength ?? 1, color_correct: !0}
    #     : delete i.img2img
    # Every community implementation (launcher, ComfyUI_RS_NAI_API_Request, aedial)
    # sends only the flat field, which the server ignores — pixel-proven by full
    # repaints at 0.05..0.16 across all of their wirings. We keep the flat field
    # (the web UI's params carry it too) and add the nested object when < 1.
    if inpaint_img2img_strength is not None:
        parameters["inpaintImg2ImgStrength"] = inpaint_img2img_strength
        if mask is not None and inpaint_img2img_strength < 1.0:
            parameters["img2img"] = {
                "strength": inpaint_img2img_strength,
                "color_correct": True,
            }

    # img2img / inpaint image. For infill, the plain `strength` key belongs to
    # action=img2img only (web UI sends it but it has no effect on the masked
    # region); keep it omitted.
    if image is not None:
        parameters["image"] = image
        if not is_infill:
            parameters["strength"] = strength
        parameters["noise"] = noise

    # inpaint mask
    if mask is not None:
        parameters["mask"] = mask

    # Vibe Transfer references (must be encodings from ai/encode-vibe for V4+)
    if reference_image_multiple:
        parameters["reference_image_multiple"] = reference_image_multiple
        parameters["reference_strength_multiple"] = reference_strength_multiple or [
            0.6
        ] * len(reference_image_multiple)
        parameters["reference_information_extracted_multiple"] = (
            reference_information_extracted_multiple
            or [1.0] * len(reference_image_multiple)
        )

    # Precise (director) reference — field-by-field from the launcher's
    # buildPreciseReferenceParameters (nai_image_request_builder.dart L349-385).
    if precise_references:
        parameters["normalize_reference_strength_multiple"] = True
        parameters["director_reference_images"] = [r["image"] for r in precise_references]
        parameters["director_reference_descriptions"] = [
            {
                "caption": {"base_caption": r["caption"], "char_captions": []},
                "legacy_uc": False,
            }
            for r in precise_references
        ]
        parameters["director_reference_information_extracted"] = [
            1 for _ in precise_references
        ]
        parameters["director_reference_strength_values"] = [
            r["strength"] for r in precise_references
        ]
        parameters["director_reference_secondary_strength_values"] = [
            r["secondary"] for r in precise_references
        ]

    # V4 structured prompt + structural flags. Reference buildV4Parameters L121-189.
    if model.is_v4:
        parameters["use_coords"] = False
        parameters["legacy_v3_extend"] = False
        parameters["legacy_uc"] = False
        parameters["characterPrompts"] = []
        if v4_prompt is not None:
            parameters["v4_prompt"] = v4_prompt
        if v4_negative_prompt is not None:
            parameters["v4_negative_prompt"] = v4_negative_prompt

    # Determine the actual model name for the API request.
    # All NAI models require "-inpainting" model variant for infill action.
    # v3: use dedicated inpaint_model enum (already has "-inpainting" suffix)
    # v4/v4.5: append "-inpainting" suffix (no separate enum variant)
    # Reference: ComfyUI-NAIDGenerator nodes.py L398
    if action is NaiAction.infill:
        if model.is_v3:
            model_name = model.inpaint_model.value
        else:
            model_name = model.value + "-inpainting"
    else:
        model_name = model.value

    request: dict[str, Any] = {
        "input": prompt,
        "model": model_name,
        "action": action.value,
        "parameters": parameters,
        # Reference: nai_image_request_builder.dart requestData L481.
        "use_new_shared_trial": True,
    }

    return request


def build_augment_request(
    image: str,
    width: int,
    height: int,
    req_type: str,
    prompt: str = "",
    defry: int = 0,
) -> dict[str, Any]:
    """Build an /ai/augment-image request body (Director Tools)."""
    return {
        "image": image,
        "width": width,
        "height": height,
        "req_type": req_type,
        "prompt": prompt,
        "defry": defry,
    }


def build_encode_vibe_request(
    image: str,
    model: NaiModel,
    information_extracted: float = 1.0,
) -> dict[str, Any]:
    """Build an /ai/encode-vibe request body."""
    return {
        "image": image,
        "model": model.value,
        "information_extracted": information_extracted,
    }


# ---------------------------------------------------------------------------
# Inpaint mask + patch compositing
#
# Ported from the reference launcher's inpaint_mask_utils.dart
# (prepareNovelAiInpaintMaskArtifacts / extractGeneratedPatch). NAI expects a
# clean binary mask aligned to its 8px latent grid — NOT Krita's ComfyUI-oriented
# feather/grow/blend mask, which makes NAI redraw a wrong-shaped region. The
# generated result is composited client-side into a transparent patch so the
# original shows through outside the mask (zero colour drift, no seam/black edge).
# ---------------------------------------------------------------------------


# Threshold LUT: coverage > 155 -> repaint (255), else 0. Matches the launcher's
# _thresholdCoverageAlpha (alpha > 155). bytes.translate applies it at C speed.
_MASK_THRESHOLD_LUT = bytes(255 if i > 155 else 0 for i in range(256))


def _mask_coverage_gray(mask: Image):
    """Return a fresh Grayscale8 QImage whose value is the mask coverage.

    Krita selection masks (Mask.to_image) are Grayscale8 with the gray value as the
    selection coverage, so a format conversion is all that is needed. The result
    always owns its data (safe to scale/reinterpret/mutate without touching the
    caller's mask)."""
    from PyQt5.QtGui import QImage as _QImage

    q = mask._qimage
    if q.format() != _QImage.Format.Format_Grayscale8:
        return q.convertToFormat(_QImage.Format.Format_Grayscale8)
    return q.copy()


def _scale_gray(gray, w: int, h: int, smooth: bool):
    """Resize a Grayscale8 QImage. Nearest (smooth=False) matches the launcher's
    latent-grid resampling; bilinear (smooth=True) keeps a soft composite edge."""
    from PyQt5.QtCore import Qt as _Qt

    if gray.width() == w and gray.height() == h:
        return gray
    mode = (
        _Qt.TransformationMode.SmoothTransformation
        if smooth
        else _Qt.TransformationMode.FastTransformation
    )
    return gray.scaled(w, h, _Qt.AspectRatioMode.IgnoreAspectRatio, mode)


def _threshold_gray(gray):
    """Binarize a Grayscale8 QImage via the >155 LUT, returning a fresh, tightly
    packed Grayscale8 QImage of 0/255. Handles padded scanlines (bytesPerLine)."""
    from PyQt5.QtGui import QImage as _QImage

    w, h = gray.width(), gray.height()
    bpl = gray.bytesPerLine()
    src = bytes(gray.constBits().asarray(bpl * h))
    out = bytearray(w * h)
    for y in range(h):
        out[y * w : y * w + w] = src[y * bpl : y * bpl + w].translate(_MASK_THRESHOLD_LUT)
    buf = bytes(out)
    # copy() -> the QImage owns its pixels so buf can be freed safely.
    return _QImage(buf, w, h, w, _QImage.Format.Format_Grayscale8).copy()


def prepare_nai_precise_reference_image(image: Image) -> Image:
    """Normalize a precise-reference image the way the NAI web client (Q6) does.

    Ported from the launcher's NAIApiUtils.ensurePngFormat: pick the standard
    canvas — (1024,1536) portrait / (1536,1024) landscape / (1472,1472) square —
    whose aspect ratio is closest to the source, aspect-preserving downscale,
    then center-paste onto a black RGB background.
    """
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtGui import QColor as _QColor, QImage as _QImage, QPainter as _QPainter

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
        nw, nh, _Qt.AspectRatioMode.IgnoreAspectRatio,
        _Qt.TransformationMode.SmoothTransformation,
    )
    canvas = _QImage(tw, th, _QImage.Format.Format_RGB32)
    canvas.fill(_QColor(0, 0, 0))
    painter = _QPainter(canvas)
    painter.drawImage((tw - nw) // 2, (th - nh) // 2, scaled)
    painter.end()
    return Image(canvas)


def prepare_nai_request_mask(mask: Image, target: Extent, latent: int = 8) -> Image:
    """Clean binary inpaint mask aligned to NAI's latent grid.

    Ported from prepareNovelAiInpaintMaskArtifacts (request-mask path): coverage ->
    nearest-downsample to target/latent -> threshold (>155) -> nearest-upscale to
    target -> RGBA white (255,255,255,255) = repaint / black (0,0,0,255) elsewhere,
    exactly what the launcher's _binaryMaskToImage sends.

    Sending this instead of Krita's soft feathered mask is what fixes NAI redrawing
    a region smaller/different than the selection with irregular black borders.
    """
    from PyQt5.QtGui import QImage as _QImage

    tw, th = target.width, target.height
    lw = max(1, tw // latent)
    lh = max(1, th // latent)

    gray = _mask_coverage_gray(mask)
    small = _scale_gray(gray, lw, lh, smooth=False)    # -> latent grid (nearest)
    binary = _threshold_gray(small)                    # >155 -> 255 else 0
    big = _scale_gray(binary, tw, th, smooth=False)    # -> target (nearest, aligned)
    # Grayscale8 0/255 -> RGBA replicates the value into R,G,B with opaque alpha:
    # white (255,255,255,255) where repaint, black (0,0,0,255) elsewhere.
    return Image(big.convertToFormat(_QImage.Format.Format_RGBA8888))


def _to_alpha8(img):
    """Return a fresh Alpha8 QImage whose alpha channel is ``img``'s gray/luminance
    value. Accepts a Grayscale8 image (gray -> alpha, zero-copy reinterpret) or an
    R=G=B RGB image (luminance -> alpha, exact because the channels are equal).

    Qt's SmoothTransformation promotes Grayscale8 to RGB32 while scaling, so the blur
    path arrives here as RGB and must be reduced back to a single 8bpp channel before
    it can be reinterpreted as alpha. Only ARGB32 images may be QPainter destinations,
    so these Alpha8 results are used strictly as compositing *sources*."""
    from PyQt5.QtGui import QImage as _QImage

    if img.format() != _QImage.Format.Format_Grayscale8:
        img = img.convertToFormat(_QImage.Format.Format_Grayscale8)
    else:
        img = img.copy()                               # detach so reinterpret is legal
    img.reinterpretAsFormat(_QImage.Format.Format_Alpha8)
    return img


def composite_nai_patch(generated: Image, mask: Image, target: Extent, feather: int = 0) -> Image:
    """Transparent RGBA patch: the generation scaled to ``target`` with alpha = the
    *binarised* selection coverage — fully opaque (alpha=255) inside the selection,
    fully transparent (alpha=0) outside.

    Ported from extractGeneratedPatch / composeGeneratedImageArtifact (patch path):
    written back as a layer, the original shows through unmasked areas (no colour drift,
    no black border) while the masked area shows the redraw.

    Why binarise (was the semi-transparency bug): Krita's ``hires_mask`` coverage is
    softly feathered, and the previous inward Qt-blur made the alpha ramp *down* toward
    the selection interior — so small selections never reached alpha=255 and every edge
    faded out (user report: "越靠边越透明, 能透出下面图层"). Thresholding the coverage the
    SAME way as the request mask (>155, see _threshold_gray / prepare_nai_request_mask)
    makes the patch solid inside and keeps the composited region identical to what NAI
    actually regenerated — so the redraw is opaque and its strength is judgeable.

    ``feather`` is retained for signature compatibility but ignored (0 = hard edge).
    A soft edge would have to be biased *outward* (launcher: dilate+blur) to avoid
    re-introducing the interior wash-out; kept out for now to guarantee opacity.
    """
    from PyQt5.QtGui import QImage as _QImage, QPainter as _QPainter

    tw, th = target.width, target.height
    gen = Image.scale(generated, target)
    q = gen._qimage
    patch = q.convertToFormat(_QImage.Format.Format_ARGB32)  # opaque RGB base
    if patch is q:                                     # already ARGB32 -> own a copy
        patch = patch.copy()

    cov = _mask_coverage_gray(mask)                    # fresh Grayscale8 coverage
    cov = _scale_gray(cov, tw, th, smooth=False)       # to target (nearest, keep 8bpp)
    binary = _threshold_gray(cov)                      # >155 -> 255 else 0 (solid interior)
    clip = _to_alpha8(binary)                          # binarised coverage as alpha

    painter = _QPainter(patch)
    painter.setCompositionMode(_QPainter.CompositionMode.CompositionMode_DestinationIn)
    painter.drawImage(0, 0, clip)                      # patch.alpha = 255 inside, 0 outside
    painter.end()
    return Image(patch)


# ---------------------------------------------------------------------------
# NAI stream endpoint (/ai/generate-image-stream) response parsing
# ---------------------------------------------------------------------------
# The stream endpoint is the one the launcher's own UI uses, and the only path
# where the masked-region strength knob (inpaintImg2ImgStrength) demonstrably
# works. The response is a sequence of frames:
#   [4-byte big-endian length][MessagePack message]
# Each message is a map: {event_type, samp_ix, step_ix, gen_id, sigma, image}.
# event_type: "intermediate" (preview) or "final"; image is PNG bytes (bin) or
# base64 (str). We buffer the whole body and extract only the "final" images.
# Pure-Python MessagePack subset decoder — Krita's Python has no msgpack module.


def _msgpack_decode(buf: bytes, pos: int):
    """Decode one MessagePack object at ``pos``; return (obj, new_pos)."""
    import struct

    b = buf[pos]
    pos += 1
    if b <= 0x7F:  # positive fixint
        return b, pos
    if b >= 0xE0:  # negative fixint
        return b - 0x100, pos
    if 0x80 <= b <= 0x8F:  # fixmap
        return _msgpack_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:  # fixarray
        return _msgpack_array(buf, pos, b & 0x0F)
    if 0xA0 <= b <= 0xBF:  # fixstr
        n = b & 0x1F
        return buf[pos : pos + n].decode("utf-8", "replace"), pos + n
    if b == 0xC0:
        return None, pos
    if b == 0xC2:
        return False, pos
    if b == 0xC3:
        return True, pos
    if b in (0xC4, 0xC5, 0xC6):  # bin8/16/32
        w = 1 << (b - 0xC4)
        n = int.from_bytes(buf[pos : pos + w], "big")
        pos += w
        return bytes(buf[pos : pos + n]), pos + n
    if b in (0xC7, 0xC8, 0xC9):  # ext8/16/32 -> skip payload, return None
        w = 1 << (b - 0xC7)
        n = int.from_bytes(buf[pos : pos + w], "big")
        pos += w + 1  # +1 ext type byte
        return None, pos + n
    if b == 0xCA:
        return struct.unpack_from(">f", buf, pos)[0], pos + 4
    if b == 0xCB:
        return struct.unpack_from(">d", buf, pos)[0], pos + 8
    if b in (0xCC, 0xCD, 0xCE, 0xCF):  # uint8/16/32/64
        w = 1 << (b - 0xCC)
        return int.from_bytes(buf[pos : pos + w], "big"), pos + w
    if b in (0xD0, 0xD1, 0xD2, 0xD3):  # int8/16/32/64
        w = 1 << (b - 0xD0)
        return int.from_bytes(buf[pos : pos + w], "big", signed=True), pos + w
    if 0xD4 <= b <= 0xD8:  # fixext1/2/4/8/16 -> skip
        n = 1 << (b - 0xD4)
        return None, pos + 1 + n  # +1 ext type byte
    if b in (0xD9, 0xDA, 0xDB):  # str8/16/32
        w = 1 << (b - 0xD9)
        n = int.from_bytes(buf[pos : pos + w], "big")
        pos += w
        return buf[pos : pos + n].decode("utf-8", "replace"), pos + n
    if b in (0xDC, 0xDD):  # array16/32
        w = 2 if b == 0xDC else 4
        n = int.from_bytes(buf[pos : pos + w], "big")
        return _msgpack_array(buf, pos + w, n)
    if b in (0xDE, 0xDF):  # map16/32
        w = 2 if b == 0xDE else 4
        n = int.from_bytes(buf[pos : pos + w], "big")
        return _msgpack_map(buf, pos + w, n)
    raise ValueError(f"msgpack: unsupported type byte 0x{b:02x} at {pos - 1}")


def _msgpack_array(buf: bytes, pos: int, n: int):
    items = []
    for _ in range(n):
        v, pos = _msgpack_decode(buf, pos)
        items.append(v)
    return items, pos


def _msgpack_map(buf: bytes, pos: int, n: int):
    result = {}
    for _ in range(n):
        k, pos = _msgpack_decode(buf, pos)
        v, pos = _msgpack_decode(buf, pos)
        if isinstance(k, bytes):
            k = k.decode("utf-8", "replace")
        result[k] = v
    return result, pos


def parse_nai_stream_response(data: bytes) -> list[bytes]:
    """Extract the final image PNGs from a buffered generate-image-stream body.

    Raises RuntimeError when the stream carries an error event. Intermediate
    (preview) frames are skipped. Results are ordered by sample index.
    """
    import base64

    finals: dict[int, bytes] = {}
    pos, size = 0, len(data)
    while pos + 4 <= size:
        frame_len = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        end = pos + frame_len
        if frame_len <= 0 or end > size:
            break  # truncated / malformed tail
        try:
            msg, _ = _msgpack_decode(data, pos)
        except Exception:
            msg = None
        pos = end
        if not isinstance(msg, dict):
            continue
        event = str(msg.get("event_type") or "")
        if event == "error" or "error" in msg:
            detail = msg.get("message") or msg.get("error") or "stream generation failed"
            raise RuntimeError(f"NAI stream error: {detail}")
        if event != "final":
            continue
        img = msg.get("image")
        if isinstance(img, str) and img:
            try:
                img = base64.b64decode(img)
            except Exception:
                continue
        if isinstance(img, (bytes, bytearray)) and len(img) > 0:
            try:
                sample = int(msg.get("samp_ix") or 0)
            except (TypeError, ValueError):
                sample = len(finals)
            finals[sample] = bytes(img)
    return [finals[k] for k in sorted(finals)]
