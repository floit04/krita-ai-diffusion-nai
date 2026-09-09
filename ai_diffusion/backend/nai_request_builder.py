"""NovelAI request construction.

Port of the launcher's `nai_image_request_builder.dart` (4.0.2, commit
ae7990c). Field names, order and conditions mirror the Dart source line for
line; the launcher in turn mirrors novelai.net web build ae6a6aa-production.
Do not add, drop or reorder fields here without checking the reference first —
this dict IS the wire contract, and every past divergence from it has shown up
as a user-visible defect (colour shift, ignored strength, "参数不全").

Differences from the Dart file, all inherent to the plugin:
- No character prompts and no prompt-edit-document markup (the plugin has
  neither), so those code paths collapse to the empty case.
- Vibe encodings arrive pre-fetched (NaiClient caches ai/encode-vibe results);
  the builder never performs network calls.
- Streaming is not offered: infill on the stream endpoint ignores the mask
  (pixel-proven 2026-07-29), and the launcher's own Krita bridge also forces
  non-stream, so `isStream` is dropped entirely.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..image import Extent, Image
from .nai_mask_utils import InpaintMaskArtifacts, prepare_inpaint_mask_artifacts
from .nai_registry import (
    ImageModels,
    QualityTags,
    UcPresets,
    build_prompt_semantics,
    capabilities_of,
)
from .nai_resolution_adapter import normalize_image_for_request

# Variety+ sigma 缩放的基准潜空间体积：4 通道 × 104 × 152（832×1216 的潜空间）。
_CFG_DELAY_REFERENCE_LATENTS = 4 * 104 * 152

# 官网当前所有图像模型共用的请求参数版本。
_OFFICIAL_PARAMS_VERSION = 4

ACTION_GENERATE = "generate"
ACTION_IMG2IMG = "img2img"
ACTION_INFILL = "infill"


def to_json_number(value: float) -> int | float:
    """The launcher's NAIApiUtils.toJsonNumber: whole floats become ints."""
    return int(value) if float(value) == int(value) else float(value)


@dataclass
class VibeReference:
    encoding: str  # pre-encoded (ai/encode-vibe) for V4+; raw base64 image for V3
    strength: float
    info_extracted: float


@dataclass
class PreciseReference:
    image_b64: str  # already normalized to a NAI reference canvas, PNG base64
    type_caption: str  # e.g. "character", the type's toApiString()
    strength: float
    fidelity: float


@dataclass
class NaiGenerationParams:
    model: str
    action: str = ACTION_GENERATE
    width: int = 1024
    height: int = 1024
    prompt: str = ""
    negative_prompt: str = ""
    scale: float = 5.0
    sampler: str = "k_euler_ancestral"
    steps: int = 28
    n_samples: int = 1
    seed: int = -1
    uc_preset: int = UcPresets.none_api_value
    quality_toggle: bool = True
    quality_tier: str = QualityTags.standard_tier
    cfg_rescale: float = 0.0
    noise_schedule: str = "karras"
    variety_plus: bool = False
    transparent_background: bool = False
    # Alpha encoding is an account-level setting on the website, independent of
    # the transparent-background tag: launcher image_params.dart:174 defaults it
    # to true, and all three of our official reference renders carry
    # straight_alpha: true - including the one with transparency switched off.
    straight_alpha: bool = True
    # img2img / infill
    source_image: Image | None = None
    mask_image: Image | None = None
    strength: float = 0.7
    noise: float = 0.0
    inpaint_strength: float = 1.0
    mask_closing_iterations: int = 0
    mask_expansion_iterations: int = 0
    # references
    vibes: list[VibeReference] = field(default_factory=list)
    precise_references: list[PreciseReference] = field(default_factory=list)


@dataclass
class NaiRequestBuildResult:
    seed: int
    effective_prompt: str
    effective_negative_prompt: str
    request_data: dict
    normalized_source: Image | None = None
    mask_artifacts: InpaintMaskArtifacts | None = None


def _resolve_noise_schedule(params: NaiGenerationParams) -> str:
    capabilities = capabilities_of(params.model)
    if not capabilities.supports_noise_schedule:
        return "karras"
    if params.noise_schedule == "native" and not capabilities.allows_native_noise_schedule:
        return "karras"
    return params.noise_schedule


def build_base_parameters(
    params: NaiGenerationParams, sampler: str, seed: int, effective_negative_prompt: str
) -> dict:
    capabilities = capabilities_of(params.model)
    noise_schedule = _resolve_noise_schedule(params)
    uses_brownian = sampler == "k_euler_ancestral" and noise_schedule != "native"
    is_v4 = capabilities.is_v4_prompt

    p: dict = {}
    p["params_version"] = _OFFICIAL_PARAMS_VERSION
    p["width"] = params.width
    p["height"] = params.height
    p["scale"] = to_json_number(params.scale)
    p["sampler"] = sampler
    p["steps"] = params.steps
    p["n_samples"] = params.n_samples
    p["ucPresetId"] = UcPresets.preset_id(params.uc_preset)
    p["qualityPresetId"] = _resolve_quality_preset_id(params)
    if is_v4:
        p["autoSmea"] = False
    p["dynamic_thresholding"] = False  # V3-only decrisp, which the plugin does not expose
    p["controlnet_strength"] = 1
    p["legacy"] = False
    p["add_original_image"] = params.action != ACTION_INFILL
    p["cfg_rescale"] = to_json_number(params.cfg_rescale)
    p["noise_schedule"] = noise_schedule
    if is_v4 or params.inpaint_strength != 1.0:
        p["inpaintImg2ImgStrength"] = to_json_number(params.inpaint_strength)
    p["seed"] = seed
    if effective_negative_prompt == "":
        p["uc"] = ""
    else:
        p["negative_prompt"] = effective_negative_prompt
    if uses_brownian:
        p["deliberate_euler_ancestral_bug"] = False
        p["prefer_brownian"] = True
    p["image_format"] = "png"

    # sigma 基数按模型取（V4.5 起 58，更早 19），再按潜空间面积缩放。
    if params.variety_plus and capabilities.supports_variety_plus:
        p["skip_cfg_above_sigma"] = capabilities.cfg_delay_sigma * math.sqrt(
            4.0 * (params.width // 8) * (params.height // 8) / _CFG_DELAY_REFERENCE_LATENTS
        )
    elif capabilities.supports_variety_plus and capabilities.retains_variety_plus:
        p["skip_cfg_above_sigma"] = None

    if capabilities.supports_transparent_background:
        # 官网把 Alpha 模式作为账号级设置，只要模型支持透明就随请求下发。
        p["straight_alpha"] = params.straight_alpha
        if params.transparent_background:
            p["tag_hint_transparent_background"] = True

    # 官网每个请求都带质量/负面预设的数字提示（0=none 1=standard 2=heavy
    # 3=light 4=humanFocus 5=furryFocus）。
    qt_hint = QualityTags.to_tag_hint(
        params.model, enabled=params.quality_toggle, tier=params.quality_tier
    )
    if qt_hint is not None:
        p["tag_hint_qt"] = qt_hint
    uc_hint = UcPresets.to_tag_hint(params.uc_preset)
    if uc_hint is not None:
        p["tag_hint_uc_preset"] = uc_hint

    if not is_v4:
        # V3 SMEA: auto above 1MP unless the sampler is ddim (launcher's
        # effectiveSmea for the plugin's fixed "auto" setting).
        is_ddim = "ddim" in sampler
        auto_smea = params.width * params.height > 1024 * 1024 and params.action == ACTION_GENERATE
        p["sm"] = False if is_ddim else auto_smea
        p["sm_dyn"] = False

    return p


def _resolve_quality_preset_id(params: NaiGenerationParams) -> str:
    if not params.quality_toggle:
        return "none"
    if params.quality_tier in QualityTags.tiers_for_model(params.model):
        return params.quality_tier
    return QualityTags.standard_tier


def build_v4_parameters(
    p: dict, params: NaiGenerationParams, effective_prompt: str, effective_negative_prompt: str
) -> None:
    p["params_version"] = _OFFICIAL_PARAMS_VERSION
    p["use_coords"] = False
    p["legacy_v3_extend"] = False
    p["legacy_uc"] = False
    p["normalize_reference_strength_multiple"] = True
    p["v4_prompt"] = {
        "caption": {"base_caption": effective_prompt, "char_captions": []},
        "use_coords": False,
        "use_order": True,
    }
    p["v4_negative_prompt"] = {
        "caption": {"base_caption": effective_negative_prompt, "char_captions": []},
        "legacy_uc": False,
    }
    p["characterPrompts"] = []


def build_vibe_transfer_parameters(p: dict, params: NaiGenerationParams) -> None:
    capabilities = capabilities_of(params.model)
    if params.precise_references:
        # Precise Reference 与 Vibe Transfer 不兼容，同时存在时保留前者。
        return
    if params.action == ACTION_INFILL:
        # infill 请求继续附带 Vibe payload 会触发服务端 500。
        return
    if not capabilities.supports_vibe_transfer:
        return
    if not params.vibes:
        return
    if capabilities.supports_encoded_vibe_transfer:
        p["normalize_reference_strength_multiple"] = True
    p["reference_image_multiple"] = [v.encoding for v in params.vibes]
    p["reference_strength_multiple"] = [v.strength for v in params.vibes]
    p["reference_information_extracted_multiple"] = [v.info_extracted for v in params.vibes]


def build_precise_reference_parameters(p: dict, params: NaiGenerationParams) -> None:
    refs = params.precise_references
    if not refs or not capabilities_of(params.model).supports_precise_reference:
        return
    p["normalize_reference_strength_multiple"] = True
    p["director_reference_images"] = [r.image_b64 for r in refs]
    p["director_reference_descriptions"] = [
        {"caption": {"base_caption": r.type_caption, "char_captions": []}, "legacy_uc": False}
        for r in refs
    ]
    p["director_reference_information_extracted"] = [1 for _ in refs]
    p["director_reference_strength_values"] = [r.strength for r in refs]
    p["director_reference_secondary_strength_values"] = [1.0 - r.fidelity for r in refs]


def map_sampler_for_model(sampler: str, model: str) -> str:
    """Coerce a sampler the model cannot run, the way the launcher does.

    Copied from nai_image_generation_api_service.dart:65-79 (mapSamplerForModel),
    which the launcher applies at :340 right before build(). novelai.net hides ddim
    for V4+ entirely, so nothing there can produce such a request.

    We need it because ddim IS reachable here: style.py:522 maps the ComfyUI "DDIM"
    preset to "ddim" during the legacy style upgrade, and a user-authored
    samplers.json preset can name it outright. Sent as-is to a V4/V5 model it also
    flips the uses_brownian gate off, dropping prefer_brownian and
    deliberate_euler_ancestral_bug that every official V5 render carries.
    """
    if sampler in ("ddim", "ddim_v3"):
        caps = capabilities_of(model)
        if caps.is_v4_prompt:
            return "k_euler_ancestral"
        if "diffusion-3" in model:
            return "ddim_v3"
    return sampler


def build_request(params: NaiGenerationParams) -> NaiRequestBuildResult:
    assert params.sampler, "Sampler cannot be empty"
    seed = params.seed if params.seed != -1 else random.randint(0, 4294967294)

    base_model = ImageModels.resolve_base_model(params.model)
    request_model = (
        ImageModels.resolve_inpainting_model(base_model)
        if params.action == ACTION_INFILL
        else params.model
    )
    semantics = build_prompt_semantics(
        params.prompt,
        params.negative_prompt,
        base_model,
        quality_toggle=params.quality_toggle,
        uc_preset=params.uc_preset,
        transparent_background=params.transparent_background,
        quality_tier=params.quality_tier,
    )
    effective_prompt = semantics.effective_prompt
    effective_negative = semantics.effective_negative_prompt

    sampler = map_sampler_for_model(params.sampler, params.model)
    p = build_base_parameters(params, sampler, seed, effective_negative)
    normalized_source: Image | None = None
    mask_artifacts: InpaintMaskArtifacts | None = None

    if capabilities_of(params.model).is_v4_prompt:
        build_v4_parameters(p, params, effective_prompt, effective_negative)

    extent = Extent(params.width, params.height)
    if params.action == ACTION_IMG2IMG and params.source_image is not None:
        normalized_source = normalize_image_for_request(params.source_image, extent)
        p["image"] = normalized_source.to_base64()
        p["strength"] = params.strength
        p["noise"] = params.noise
        p["color_correct"] = False

    if (
        params.action == ACTION_INFILL
        and params.source_image is not None
        and params.mask_image is not None
    ):
        mask_artifacts = prepare_inpaint_mask_artifacts(
            params.mask_image,
            extent,
            closing_iterations=params.mask_closing_iterations,
            expansion_iterations=params.mask_expansion_iterations,
        )
        normalized_source = normalize_image_for_request(params.source_image, extent)
        p["image"] = normalized_source.to_base64()
        p["mask"] = mask_artifacts.request_mask.to_base64()
        p["strength"] = to_json_number(params.strength)
        p["noise"] = to_json_number(params.noise)
        if (
            ImageModels.supports_img2img_inpainting(request_model)
            and params.inpaint_strength != 1.0
        ):
            p["img2img"] = {
                "strength": to_json_number(params.inpaint_strength),
                "color_correct": True,
            }

    if "image" in p:
        # 官网对带底图的请求固定 extra_noise_seed = seed - 1，缺发时服务端自选
        # 加噪种子，同种子无法复现。
        p["extra_noise_seed"] = seed - 1

    build_vibe_transfer_parameters(p, params)
    build_precise_reference_parameters(p, params)

    request_data = {
        "input": effective_prompt,
        "model": request_model,
        "action": params.action,
        "parameters": p,
        "use_new_shared_trial": True,
    }
    return NaiRequestBuildResult(
        seed=seed,
        effective_prompt=effective_prompt,
        effective_negative_prompt=effective_negative,
        request_data=request_data,
        normalized_source=normalized_source,
        mask_artifacts=mask_artifacts,
    )
