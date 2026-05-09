"""NovelAI image generation parameter preparation.

Converts plugin UI state into NAI API request bodies.
This module is independent from the ComfyUI workflow system.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .image import Extent, Image, multiple_of
from .settings import ImageFileFormat
from .util import client_logger as log


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
        return NaiSampler.k_euler


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
        return NaiNoiseSchedule.native


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

# Maximum total pixel count for NAI (roughly)
_NAI_MAX_PIXELS = 1048576  # 1024*1024


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
    # Variety+ (skip_cfg_above_sigma)
    skip_cfg_above_sigma: float | None = None,
    # img2img / inpaint
    image: str | None = None,
    strength: float = 0.7,
    noise: float = 0.0,
    mask: str | None = None,
    # Vibe Transfer
    reference_image_multiple: list[str] | None = None,
    reference_strength_multiple: list[float] | None = None,
    reference_information_extracted_multiple: list[float] | None = None,
    # Output format
    image_format: str = "png",
    # V4 structured prompt (optional)
    v4_prompt: dict | None = None,
    v4_negative_prompt: dict | None = None,
) -> dict[str, Any]:
    """Build the full NAI /ai/generate-image request body."""

    parameters: dict[str, Any] = {
        "width": width,
        "height": height,
        "sampler": sampler.value,
        "steps": steps,
        "scale": scale,
        "cfg_rescale": cfg_rescale,
        "noise_schedule": noise_schedule.value,
        "seed": seed,
        "n_samples": n_samples,
        "qualityToggle": quality_toggle,
        "ucPreset": uc_preset.value,
        "sm": False,
        "sm_dyn": False,
        "dynamic_thresholding": dynamic_thresholding,
        "negative_prompt": negative_prompt,
        "image_format": image_format,
        "params_version": 3,
    }

    # Variety+ — skip CFG above a sigma threshold
    if skip_cfg_above_sigma is not None:
        parameters["skip_cfg_above_sigma"] = skip_cfg_above_sigma

    # img2img / inpaint image
    if image is not None:
        parameters["image"] = image
        parameters["strength"] = strength
        parameters["noise"] = noise
        parameters["extra_noise_seed"] = seed

    # inpaint mask
    if mask is not None:
        parameters["mask"] = mask
        parameters["add_original_image"] = True

    # Vibe Transfer references
    if reference_image_multiple:
        parameters["reference_image_multiple"] = reference_image_multiple
        parameters["reference_strength_multiple"] = reference_strength_multiple or [
            0.6
        ] * len(reference_image_multiple)
        parameters["reference_information_extracted_multiple"] = (
            reference_information_extracted_multiple
            or [1.0] * len(reference_image_multiple)
        )

    # V4 structured prompt
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
