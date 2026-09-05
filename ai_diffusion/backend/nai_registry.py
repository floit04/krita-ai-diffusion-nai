"""Model capability registry and prompt semantics for the NovelAI API.

Straight port of the NAI launcher's `model_capabilities.dart`,
`api_constants.dart` (QualityTags / UcPresets / ImageModels) and
`prompt_semantics_utils.dart` + `novelai_auto_text.dart`, version 4.0.2
(commit ae7990c, 2026-09-05). Structure and naming mirror the Dart source so
the two can be diffed side by side; do not "improve" values here — every
string and number is part of the wire contract with novelai.net.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

# ---------------------------------------------------------------------------
# ImageModels (api_constants.dart)
# ---------------------------------------------------------------------------


class ImageModels:
    v3 = "nai-diffusion-3"
    v3_inpainting = "nai-diffusion-3-inpainting"
    furry_v3 = "nai-diffusion-furry-3"
    furry_v3_inpainting = "nai-diffusion-furry-3-inpainting"
    v4_curated = "nai-diffusion-4-curated-preview"
    v4_full = "nai-diffusion-4-full"
    v4_curated_inpainting = "nai-diffusion-4-curated-inpainting"
    v4_full_inpainting = "nai-diffusion-4-full-inpainting"
    v45_curated = "nai-diffusion-4-5-curated"
    v45_curated_inpainting = "nai-diffusion-4-5-curated-inpainting"
    v45_full = "nai-diffusion-4-5-full"
    v45_full_inpainting = "nai-diffusion-4-5-full-inpainting"
    v5_curated = "nai-diffusion-5-curated"
    v5_curated_inpainting = "nai-diffusion-5-curated-inpainting"
    v5_full = "nai-diffusion-5-full"
    v5_full_inpainting = "nai-diffusion-5-full-inpainting"
    v5_staging_key = "custom"

    _inpainting_to_base: ClassVar = {
        v5_full_inpainting: v5_full,
        v5_curated_inpainting: v5_curated,
        v45_full_inpainting: v45_full,
        v45_curated_inpainting: v45_curated,
        v4_full_inpainting: v4_full,
        v4_curated_inpainting: v4_curated,
        v3_inpainting: v3,
        furry_v3_inpainting: furry_v3,
    }

    @classmethod
    def migrate_legacy_model(cls, model: str) -> str:
        return cls.v5_curated if model == cls.v5_staging_key else model

    @classmethod
    def is_inpainting_model(cls, model: str) -> bool:
        return model in cls._inpainting_to_base

    @classmethod
    def resolve_inpainting_model(cls, raw_model: str) -> str:
        model = cls.migrate_legacy_model(raw_model)
        if cls.is_inpainting_model(model):
            return model
        if not capabilities_of(model).has_inpainting_variant:
            return model
        return {
            # V5 Curated 的重绘权重尚未就绪，网页端映射到 V4.5 Curated Inpainting。
            cls.v5_curated: cls.v45_curated_inpainting,
            cls.v5_full: cls.v5_full_inpainting,
            cls.v45_full: cls.v45_full_inpainting,
            cls.v45_curated: cls.v45_curated_inpainting,
            cls.v4_full: cls.v4_full_inpainting,
            cls.v4_curated: cls.v4_curated_inpainting,
            cls.furry_v3: cls.furry_v3_inpainting,
        }.get(model, cls.v3_inpainting)

    @classmethod
    def resolve_base_model(cls, model: str) -> str:
        return cls._inpainting_to_base.get(model, model)

    @classmethod
    def supports_img2img_inpainting(cls, model: str) -> bool:
        return capabilities_of(model).supports_img2img_inpainting


# ---------------------------------------------------------------------------
# ModelCapabilities (model_capabilities.dart)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapabilities:
    id: str
    prompt_structure: str = "legacy"  # "legacy" | "v4"
    token_limit: int = 225
    params_version: int = 3
    default_scale: float = 5.0
    default_steps: int = 23
    max_characters: int = 0
    supports_vibe_transfer: bool = False
    supports_encoded_vibe_transfer: bool = False
    supports_precise_reference: bool = False
    has_inpainting_variant: bool = True
    supports_img2img_inpainting: bool = False
    supports_transparent_background: bool = False
    supports_enhance_prompt_add: bool = False
    supports_text_rendering: bool = False
    supports_auto_text: bool = False
    supports_noise_schedule: bool = True
    supports_variety_plus: bool = False
    retains_variety_plus: bool = True
    cfg_delay_sigma: float = 19.0

    @property
    def is_v4_prompt(self) -> bool:
        return self.prompt_structure == "v4"

    @property
    def allows_native_noise_schedule(self) -> bool:
        return self.prompt_structure == "legacy"


_V3 = ModelCapabilities(
    id=ImageModels.v3,
    supports_vibe_transfer=True,
    supports_img2img_inpainting=True,
    supports_variety_plus=True,
)
_FURRY_V3 = ModelCapabilities(
    id=ImageModels.furry_v3,
    default_scale=6.2,
    supports_vibe_transfer=True,
    supports_img2img_inpainting=True,
    supports_variety_plus=True,
)
_V4_COMMON: dict = dict(  # noqa: C408
    prompt_structure="v4",
    token_limit=512,
    params_version=4,
    max_characters=6,
    supports_vibe_transfer=True,
    supports_encoded_vibe_transfer=True,
    supports_img2img_inpainting=True,
    supports_text_rendering=True,
    supports_variety_plus=True,
)
_V4_CURATED = ModelCapabilities(id=ImageModels.v4_curated, default_scale=5.5, **_V4_COMMON)
_V4_FULL = ModelCapabilities(id=ImageModels.v4_full, default_scale=5.5, **_V4_COMMON)
_V45_CURATED = ModelCapabilities(
    id=ImageModels.v45_curated,
    default_scale=5.0,
    cfg_delay_sigma=58.0,
    supports_precise_reference=True,
    **_V4_COMMON,
)
_V45_FULL = ModelCapabilities(
    id=ImageModels.v45_full,
    default_scale=5.0,
    cfg_delay_sigma=58.0,
    supports_precise_reference=True,
    supports_enhance_prompt_add=True,
    **_V4_COMMON,
)
_V5_COMMON: dict = dict(  # noqa: C408
    prompt_structure="v4",
    params_version=4,
    default_scale=4.0,
    default_steps=28,
    max_characters=22,
    supports_img2img_inpainting=True,
    supports_transparent_background=True,
    supports_enhance_prompt_add=True,
    supports_text_rendering=True,
    supports_auto_text=True,
    # 网页端对 V5 隐藏了噪声调度与 Variety+，这里刻意放开供手动尝试。
    supports_noise_schedule=True,
    supports_variety_plus=True,
    retains_variety_plus=False,
    cfg_delay_sigma=58.0,
)
_V5_CURATED = ModelCapabilities(id=ImageModels.v5_curated, token_limit=703, **_V5_COMMON)
_V5_FULL = ModelCapabilities(id=ImageModels.v5_full, token_limit=1471, **_V5_COMMON)

_EXACT_CAPABILITIES = {
    ImageModels.v3: _V3,
    ImageModels.v3_inpainting: _V3,
    ImageModels.furry_v3: _FURRY_V3,
    ImageModels.furry_v3_inpainting: _FURRY_V3,
    ImageModels.v4_curated: _V4_CURATED,
    ImageModels.v4_curated_inpainting: _V4_CURATED,
    ImageModels.v4_full: _V4_FULL,
    ImageModels.v4_full_inpainting: _V4_FULL,
    ImageModels.v45_curated: _V45_CURATED,
    ImageModels.v45_curated_inpainting: _V45_CURATED,
    ImageModels.v45_full: _V45_FULL,
    ImageModels.v45_full_inpainting: _V45_FULL,
    ImageModels.v5_curated: _V5_CURATED,
    ImageModels.v5_curated_inpainting: _V5_CURATED,
    ImageModels.v5_full: _V5_FULL,
    ImageModels.v5_full_inpainting: _V5_FULL,
    ImageModels.v5_staging_key: _V5_CURATED,
}


def capabilities_of(model: str) -> ModelCapabilities:
    """Exact match first, then the launcher's longest-substring fallback."""
    exact = _EXACT_CAPABILITIES.get(model)
    if exact is not None:
        return exact
    if "diffusion-5" in model:
        return _V5_FULL if "full" in model else _V5_CURATED
    if "diffusion-4-5" in model:
        return _V45_FULL if "full" in model else _V45_CURATED
    if "diffusion-4" in model:
        return _V4_FULL if "full" in model else _V4_CURATED
    if "diffusion-furry-3" in model:
        return _FURRY_V3
    if "diffusion-3" in model:
        return _V3
    return _V3


# ---------------------------------------------------------------------------
# QualityTags (api_constants.dart)
# ---------------------------------------------------------------------------


class QualityTags:
    transparent_background_tag = "transparent background"

    standard_tier = "standard"
    light_tier = "light"

    _v5_tiers: ClassVar = {
        standard_tier: "very aesthetic, masterpiece, no text",
        light_tier: "very aesthetic, amazing quality, no text",
    }

    model_quality_tags: ClassVar = {
        ImageModels.v5_staging_key: "very aesthetic, masterpiece, no text",
        ImageModels.v5_full: "very aesthetic, masterpiece, no text",
        ImageModels.v5_curated: "very aesthetic, masterpiece, no text",
        ImageModels.v45_full: "location, very aesthetic, masterpiece, no text",
        ImageModels.v45_curated: "location, masterpiece, no text, -0.8::feet::, rating:general",
        ImageModels.v4_full: "no text, best quality, very aesthetic, absurdres",
        ImageModels.v4_curated: "rating:general, amazing quality, very aesthetic, absurdres",
        ImageModels.v3: "best quality, amazing quality, very aesthetic, absurdres",
        ImageModels.furry_v3: "{best quality}, {amazing quality}",
    }

    model_quality_tag_tiers: ClassVar = {
        ImageModels.v5_staging_key: _v5_tiers,
        ImageModels.v5_full: _v5_tiers,
        ImageModels.v5_curated: _v5_tiers,
    }

    # 官网正则：标记前的分隔符也算在匹配内，`text::` 是转义写法不算标记。
    text_render_marker = re.compile(r"(?:^|\s|[,.:\[\]{}、。])text:(?!:)", re.IGNORECASE)

    prompt_mix_separator = "|"
    max_prompt_mix_chunks = 6

    _prompt_mix_weight = re.compile(r":[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

    @classmethod
    def tiers_for_model(cls, model: str) -> list[str]:
        tiers = cls.model_quality_tag_tiers.get(model)
        if tiers is not None:
            return list(tiers.keys())
        return [cls.standard_tier]

    @classmethod
    def to_tag_hint(cls, model: str, enabled: bool, tier: str, omit: bool = False) -> int | None:
        if omit:
            return None
        if not enabled:
            return 0
        supports_light = cls.light_tier in cls.tiers_for_model(model)
        return 3 if supports_light and tier == cls.light_tier else 1

    @classmethod
    def get_quality_tags_for_tier(cls, model: str, tier: str) -> str | None:
        tiers = cls.model_quality_tag_tiers.get(model)
        if tiers is not None:
            return tiers.get(tier, tiers[cls.standard_tier])
        return cls.model_quality_tags.get(model)

    @classmethod
    def split_prompt_mix_chunks(cls, prompt: str) -> list[str]:
        """Split on `|`, treating `||…||` ranges as escaped, capped at 6 chunks."""
        chunks: list[str] = []
        buffer: list[str] = []
        escaped = False
        i = 0
        while i < len(prompt):
            char = prompt[i]
            if (
                char == cls.prompt_mix_separator
                and prompt[i + 1 : i + 2] == cls.prompt_mix_separator
            ):
                escaped = not escaped
                buffer.append(cls.prompt_mix_separator * 2)
                i += 2
                continue
            if char == cls.prompt_mix_separator and not escaped:
                chunks.append("".join(buffer))
                buffer.clear()
                i += 1
                continue
            buffer.append(char)
            i += 1
        chunks.append("".join(buffer))
        if len(chunks) <= cls.max_prompt_mix_chunks:
            return chunks
        head = chunks[: cls.max_prompt_mix_chunks - 1]
        tail = cls.prompt_mix_separator.join(chunks[cls.max_prompt_mix_chunks - 1 :])
        return head + [tail]

    @classmethod
    def _first_text_marker_outside_randomizer(cls, prompt: str):
        randomizer_open = False
        cursor = 0
        for match in cls.text_render_marker.finditer(prompt):
            while cursor < match.start():
                if (
                    prompt[cursor] == cls.prompt_mix_separator
                    and prompt[cursor + 1 : cursor + 2] == cls.prompt_mix_separator
                ):
                    randomizer_open = not randomizer_open
                    cursor += 2
                    continue
                cursor += 1
            if not randomizer_open:
                return match
        return None

    @classmethod
    def compose_suffix(
        cls,
        model: str,
        quality_toggle: bool,
        transparent_background: bool = False,
        quality_tier: str = standard_tier,
    ) -> str | None:
        tags = cls.get_quality_tags_for_tier(model, quality_tier) if quality_toggle else None
        has_tags = bool(tags)
        if not transparent_background:
            return tags if has_tags else None
        if has_tags:
            return f"{cls.transparent_background_tag}, {tags}"
        return cls.transparent_background_tag

    @classmethod
    def append_suffix(cls, prompt: str, suffix: str | None) -> str:
        if not suffix:
            return prompt
        trimmed = prompt.strip()
        if not trimmed:
            return suffix
        if trimmed.endswith(","):
            return f"{trimmed} {suffix}"
        return f"{trimmed}, {suffix}"

    @classmethod
    def _apply_to_chunk(cls, chunk: str, suffix: str, has_text_section: bool) -> str:
        if not has_text_section:
            return cls.append_suffix(chunk, suffix)
        match = cls._first_text_marker_outside_randomizer(chunk)
        if match is None:
            return cls.append_suffix(chunk, suffix)
        marker_and_text = chunk[match.start() :]
        needs_separator = match.group(0).lower() == "text:"
        return (
            cls.append_suffix(chunk[: match.start()], suffix)
            + (" " if needs_separator else "")
            + marker_and_text
        )

    @classmethod
    def apply_suffix(cls, prompt: str, suffix: str | None, capabilities: ModelCapabilities) -> str:
        """V4+: append only to the first mix chunk, before any `text:` marker.
        V3 and earlier: append to every `|` chunk, preserving trailing weights."""
        if not suffix:
            return prompt
        if capabilities.is_v4_prompt:
            chunks = cls.split_prompt_mix_chunks(prompt)
            chunks[0] = cls._apply_to_chunk(
                chunks[0], suffix, has_text_section=capabilities.supports_text_rendering
            )
            return cls.prompt_mix_separator.join(chunks)
        parts = []
        for chunk in prompt.split(cls.prompt_mix_separator):
            m = cls._prompt_mix_weight.search(chunk)
            weight = m.group(0) if m else ""
            base = chunk[: len(chunk) - len(weight)] if weight else chunk
            parts.append(cls.append_suffix(base, suffix) + weight)
        return cls.prompt_mix_separator.join(parts)


# ---------------------------------------------------------------------------
# UcPresets (api_constants.dart)
# ---------------------------------------------------------------------------


class UcPresets:
    heavy_api_value = 0
    light_api_value = 1
    human_focus_api_value = 2
    none_api_value = 3
    furry_focus_api_value = 7  # legacy UCPresets.furryFocus

    _FURRY_FOCUS_TEXT = (
        "{worst quality}, distracting watermark, unfinished, bad quality, {widescreen}, upscale,"
        " {sequence}, {{grandfathered content}}, blurred foreground, chromatic aberration, sketch,"
        " everyone, [sketch background], simple, [flat colors], ych (character), outline,"
        " multiple scenes, [[horror (theme)]], comic"
    )
    _V3_STYLE_FURRY = (
        "{{worst quality}}, [displeasing], {unusual pupils}, guide lines, {{unfinished}}, {bad},"
        " url, artist name, {{tall image}}, mosaic, {sketch page}, comic panel, impact (font),"
        " [dated], {logo}, ych, {what}, {where is your god now}, {distorted text}, repeated text,"
        " {floating head}, {1994}, {widescreen}, absolutely everyone, sequence,"
        " {compression artifacts}, hard translated, {cropped}, {commissioner name}, unknown text,"
        " high contrast"
    )

    v5_presets: ClassVar = {
        "heavy": "lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page",
        "light": "lowres, bad hands, bad anatomy, artistic error, sepia, white haze, worst quality, very displeasing, jpeg artifacts, 0::ai-generated::",
        "furryFocus": _FURRY_FOCUS_TEXT,
        "humanFocus": "lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page, @_@, mismatched pupils, glowing eyes, bad anatomy",
        "none": "",
    }
    v45_full_presets: ClassVar = {
        "heavy": "lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page",
        "light": "lowres, artistic error, scan artifacts, worst quality, bad quality, jpeg artifacts, multiple views, very displeasing, too many watermarks, negative space, blank page",
        "furryFocus": _FURRY_FOCUS_TEXT,
        "humanFocus": "lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page, @_@, mismatched pupils, glowing eyes, bad anatomy",
        "none": "",
    }
    v45_curated_presets: ClassVar = {
        "heavy": "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, logo, too many watermarks, negative space, blank page",
        "light": "blurry, lowres, upscaled, artistic error, scan artifacts, jpeg artifacts, logo, too many watermarks, negative space, blank page",
        "furryFocus": _FURRY_FOCUS_TEXT,
        "humanFocus": "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, bad anatomy, bad hands, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, logo, too many watermarks, @_@, mismatched pupils, glowing eyes, negative space, blank page",
        "none": "",
    }
    v4_full_presets: ClassVar = {
        "heavy": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks",
        "light": "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing",
        "furryFocus": _V3_STYLE_FURRY,
        "humanFocus": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks, bad anatomy, bad hands",
        "none": "",
    }
    v4_curated_presets: ClassVar = {
        "heavy": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts",
        "light": "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature",
        "furryFocus": _V3_STYLE_FURRY,
        "humanFocus": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts, bad anatomy, bad hands",
        "none": "",
    }
    v3_presets: ClassVar = {
        "heavy": "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        "light": "lowres, jpeg artifacts, worst quality, watermark, blurry, very displeasing",
        "furryFocus": _V3_STYLE_FURRY,
        "humanFocus": "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands, @_@, mismatched pupils, heart-shaped pupils, glowing eyes",
        "none": "lowres",
    }
    furry_v3_presets: ClassVar = {
        "heavy": _V3_STYLE_FURRY,
        "light": "{worst quality}, guide lines, unfinished, bad, url, tall image, widescreen, compression artifacts, unknown text",
        "furryFocus": _V3_STYLE_FURRY,
        "humanFocus": _V3_STYLE_FURRY,
        "none": "",
    }

    # 历史启动器版本写入过的 UC 文本，只用于识别并剥离旧数据（含 nsfw 前缀的变体）。
    legacy_preset_variants: ClassVar = {
        ImageModels.v45_full: {
            "heavy": [
                "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page"
            ],
            "light": [
                "nsfw, lowres, artistic error, scan artifacts, worst quality, bad quality, jpeg artifacts, multiple views, very displeasing, too many watermarks, negative space, blank page"
            ],
        },
    }

    _NSFW_PATTERN = re.compile(r"[\{\[]*nsfw[\}\]]*\s*,?\s*", re.IGNORECASE)
    _NSFW_CONTAINS = re.compile(r"[\{\[]*nsfw[\}\]]*", re.IGNORECASE)

    @classmethod
    def _presets_for_model(cls, model: str) -> dict[str, str]:
        if model in (
            ImageModels.v5_staging_key,
            ImageModels.v5_curated,
            ImageModels.v5_curated_inpainting,
            ImageModels.v5_full,
            ImageModels.v5_full_inpainting,
        ):
            return cls.v5_presets
        return {
            ImageModels.v45_full: cls.v45_full_presets,
            ImageModels.v45_curated: cls.v45_curated_presets,
            ImageModels.v4_full: cls.v4_full_presets,
            ImageModels.v4_curated: cls.v4_curated_presets,
            ImageModels.furry_v3: cls.furry_v3_presets,
        }.get(model, cls.v3_presets)

    @classmethod
    def preset_key_from_int(cls, uc_preset: int) -> str:
        return {
            cls.heavy_api_value: "heavy",
            cls.light_api_value: "light",
            cls.human_focus_api_value: "humanFocus",
            cls.furry_focus_api_value: "furryFocus",
        }.get(uc_preset, "none")

    @classmethod
    def preset_id(cls, uc_preset: int) -> str:
        """Value of the request's `ucPresetId` field ('heavy' | 'light' | …)."""
        return {
            cls.heavy_api_value: "heavy",
            cls.light_api_value: "light",
            cls.human_focus_api_value: "humanFocus",
            cls.none_api_value: "none",
            cls.furry_focus_api_value: "furryFocus",
        }.get(uc_preset, "none")

    @classmethod
    def to_tag_hint(cls, api_value: int, omit: bool = False) -> int | None:
        if omit:
            return None
        return {
            cls.heavy_api_value: 2,
            cls.light_api_value: 3,
            cls.human_focus_api_value: 4,
            cls.none_api_value: 0,
            cls.furry_focus_api_value: 5,
        }.get(api_value)

    @classmethod
    def get_preset_content(cls, model: str, uc_preset: int) -> str:
        key = cls.preset_key_from_int(uc_preset)
        return cls._presets_for_model(model).get(key, "")

    @staticmethod
    def _split_tags(prompt: str) -> list[str]:
        return [tag.strip() for tag in prompt.split(",") if tag.strip()]

    @classmethod
    def _strip_preset_content(cls, negative: str, preset_content: str) -> str | None:
        preset = preset_content.strip()
        if not preset:
            return None
        prompt_tags = cls._split_tags(negative)
        preset_tags = cls._split_tags(preset)
        if len(prompt_tags) < len(preset_tags):
            return None
        for ours, theirs in zip(prompt_tags, preset_tags):
            if ours.lower() != theirs.lower():
                return None
        return ", ".join(prompt_tags[len(preset_tags) :])

    @classmethod
    def strip_preset(cls, negative: str, model: str, uc_preset: int) -> str:
        trimmed = negative.strip()
        key = cls.preset_key_from_int(uc_preset)
        if not trimmed or key == "none":
            return trimmed
        current = cls.get_preset_content(model, uc_preset)
        variants = [current] if current else []
        for legacy in cls.legacy_preset_variants.get(model, {}).get(key, []):
            if legacy != current:
                variants.append(legacy)
        for content in variants:
            stripped = cls._strip_preset_content(trimmed, content)
            if stripped is not None:
                return stripped
        return trimmed

    @classmethod
    def apply_preset(cls, negative: str, model: str, uc_preset: int) -> str:
        if cls.preset_key_from_int(uc_preset) == "none":
            return negative
        preset_content = cls.get_preset_content(model, uc_preset)
        if not preset_content:
            return negative
        trimmed = cls.strip_preset(negative, model, uc_preset)
        if not trimmed:
            return preset_content
        return f"{preset_content}, {trimmed}"

    @classmethod
    def remove_nsfw_tag(cls, prompt: str) -> str:
        if not prompt:
            return prompt
        result = cls._NSFW_PATTERN.sub("", prompt)
        result = re.sub(r",\s*,", ",", result)
        result = re.sub(r"^\s*,\s*", "", result)
        result = re.sub(r"\s*,\s*$", "", result)
        return result.strip()

    @classmethod
    def contains_nsfw_tag(cls, prompt: str) -> bool:
        return bool(cls._NSFW_CONTAINS.search(prompt))

    @classmethod
    def apply_preset_with_nsfw_check(
        cls, negative: str, positive: str, model: str, uc_preset: int
    ) -> str:
        effective = cls.apply_preset(negative, model, uc_preset)
        if cls.contains_nsfw_tag(positive):
            effective = cls.remove_nsfw_tag(effective)
        return effective


# ---------------------------------------------------------------------------
# NovelAiAutoText (novelai_auto_text.dart) — V5 quote-to-`teXt:` pass
# ---------------------------------------------------------------------------


class NovelAiAutoText:
    marker = "teXt:"

    _quote_pairs: ClassVar = {'"': '"', "“": "”", "「": "」", "'": "'", "‘": "’"}
    _single_quote_boundary = re.compile(r"[\s,.]")
    _letter_or_number = re.compile(r"[^\W_]", re.UNICODE)
    _cjk = re.compile("[　-〿぀-ゟ゠-ヿ＀-ﾟ一-龯㐀-䶿]")

    @classmethod
    def _extract_quoted_texts(cls, prompt: str) -> list[str]:
        result: list[str] = []
        cursor = 0
        while cursor < len(prompt):
            opening = prompt[cursor]
            closing = cls._quote_pairs.get(opening)
            accepts = closing is not None and (
                opening != "'" or cls._is_single_quote_boundary(prompt, cursor - 1)
            )
            if not accepts:
                cursor += 1
                continue
            apostrophe_style = closing in ("'", "’")
            end = cursor + 1
            while end < len(prompt) and (
                prompt[end] != closing
                or (apostrophe_style and cls._is_letter_or_number(prompt, end + 1))
            ):
                end += 1
            if end >= len(prompt):
                cursor += 1
                continue
            value = prompt[cursor + 1 : end].strip()
            if value:
                result.append(value)
            cursor = end + 1
        return result

    @classmethod
    def _is_single_quote_boundary(cls, text: str, index: int) -> bool:
        if index < 0 or index >= len(text):
            return True
        return bool(cls._single_quote_boundary.match(text[index]))

    @classmethod
    def _is_letter_or_number(cls, text: str, index: int) -> bool:
        if index < 0 or index >= len(text):
            return False
        return bool(cls._letter_or_number.match(text[index]))

    @classmethod
    def build_block(cls, prompt: str) -> str | None:
        if QualityTags.text_render_marker.search(prompt):
            return None
        chunks = QualityTags.split_prompt_mix_chunks(prompt)
        quoted = cls._extract_quoted_texts(chunks[0])
        if not quoted:
            return None
        combined = "".join(quoted)
        cjk_count = len(cls._cjk.findall(combined))
        if cjk_count > 0 and cjk_count / len(combined) > 0.3:
            quoted = list(reversed(quoted))
        return f"{cls.marker} " + "\n\n".join(quoted)

    @classmethod
    def apply(cls, prompt: str) -> str:
        block = cls.build_block(prompt)
        if block is None:
            return prompt
        chunks = QualityTags.split_prompt_mix_chunks(prompt)
        base = re.sub(r"[\s,]+$", "", chunks[0])
        chunks[0] = block if not base else f"{base}, {block}"
        return QualityTags.prompt_mix_separator.join(chunks)


# ---------------------------------------------------------------------------
# Prompt semantics snapshot (prompt_semantics_utils.dart)
# ---------------------------------------------------------------------------


@dataclass
class PromptSemantics:
    effective_prompt: str
    effective_negative_prompt: str


def build_prompt_semantics(
    prompt: str,
    negative_prompt: str,
    model: str,
    quality_toggle: bool,
    uc_preset: int,
    transparent_background: bool = False,
    quality_tier: str = QualityTags.standard_tier,
) -> PromptSemantics:
    """The launcher's buildPromptSemanticsSnapshot, minus its edit-document
    markup and character prompts (this plugin has neither)."""
    capabilities = capabilities_of(model)
    effective_prompt = QualityTags.apply_suffix(
        prompt,
        QualityTags.compose_suffix(
            model,
            quality_toggle=quality_toggle,
            transparent_background=(
                transparent_background and capabilities.supports_transparent_background
            ),
            quality_tier=quality_tier,
        ),
        capabilities,
    )
    if capabilities.supports_auto_text:
        effective_prompt = NovelAiAutoText.apply(effective_prompt)
    effective_negative = UcPresets.apply_preset_with_nsfw_check(
        negative_prompt, prompt, model, uc_preset
    )
    return PromptSemantics(effective_prompt, effective_negative)
