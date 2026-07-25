from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from PyQt5.QtCore import QObject, pyqtSignal

from .localization import translate as _
from .platform_tools import is_macos, is_windows
from .util import client_logger as log
from .util import encode_json, read_json_with_comments, user_data_dir


class ServerMode(Enum):
    undefined = -1
    managed = 0
    external = 1
    cloud = 2
    novelai = 3


class ServerBackend(Enum):
    cpu = (_("Run on CPU"), True)
    cuda = (_("Use CUDA (NVIDIA GPU)"), not is_macos)
    mps = (_("Use MPS (Metal Performance Shader)"), is_macos)
    directml = (_("Use DirectML (GPU)"), is_windows)
    xpu = (_("Use XPU (Intel GPU)"), not is_macos)
    rocm = (_("Use ROCm (AMD GPU)"), not is_macos)

    @staticmethod
    def supported():
        return [b for b in ServerBackend if b.value[1]]

    @staticmethod
    def default():
        if is_macos:
            return ServerBackend.mps
        else:
            return ServerBackend.cuda


class GenerationFinishedAction(Enum):
    none = _("Do Nothing")
    preview = _("Preview")
    apply = _("Apply")


class ApplyBehavior(Enum):
    replace = _("Modify active layer")
    layer = _("New layer on top")
    layer_active = _("New layer above active")


class ApplyRegionBehavior(Enum):
    none = _("Do not update regions")
    replace = _("Modify region layers")
    layer_group = _("Layer group")
    transparency_mask = _("Layer group + mask")
    no_hide = _("Layer group (don't hide)")


class PerformancePreset(Enum):
    auto = _("Automatic")
    cpu = _("CPU")
    low = _("GPU low (up to 6GB)")
    medium = _("GPU medium (6GB to 12GB)")
    high = _("GPU high (more than 12GB)")
    cloud = _("Cloud")
    custom = _("Custom")


class ImageFileFormat(Enum):
    png = "PNG (fast)"  # fast, large files
    png_small = "PNG"  # slow, smaller files
    webp = "WebP"
    webp_lossless = "WebP (lossless)"
    jpeg = "JPEG"

    @staticmethod
    def from_extension(filepath: str | Path):
        extension = Path(filepath).suffix.lower()
        if extension == ".png":
            return ImageFileFormat.png_small
        if extension == ".webp":
            return ImageFileFormat.webp
        if extension in {".jpg", ".jpeg"}:
            return ImageFileFormat.jpeg
        raise ValueError(f"Unsupported image extension: {extension}")

    @property
    def extension(self):
        if self in [ImageFileFormat.png, ImageFileFormat.png_small]:
            return "png"
        elif self in [ImageFileFormat.webp, ImageFileFormat.webp_lossless]:
            return "webp"
        else:
            return "jpg"

    @property
    def quality(self):
        if self in [ImageFileFormat.png]:
            return 85
        elif self in [ImageFileFormat.png_small]:
            return 50
        elif self in [ImageFileFormat.webp]:
            return 80
        elif self in [ImageFileFormat.webp_lossless]:
            return 100
        elif self in [ImageFileFormat.jpeg]:
            return 85
        else:
            return 85

    @property
    def no_webp_fallback(self):
        if self is ImageFileFormat.webp_lossless:
            return ImageFileFormat.png
        if self is ImageFileFormat.webp:
            return ImageFileFormat.jpeg
        return self


class PerformancePresetSettings(NamedTuple):
    batch_size: int = 4
    resolution_multiplier: float = 1.0
    max_pixel_count: int = 6
    tiled_vae: bool = False


@dataclass
class PerformanceSettings:
    batch_size: int = 4
    resolution_multiplier: float = 1.0
    max_pixel_count: int = 6
    dynamic_caching: bool = False
    tiled_vae: bool = False


class Setting:
    def __init__(self, name: str, default, desc="", help="", items=None):
        self.name = name
        self.desc = desc
        self.default = default
        self.help = help
        self.items = items

    def str_to_enum(self, s: str):
        assert isinstance(self.default, Enum)
        EnumType = type(self.default)
        try:
            return EnumType[s]
        except KeyError:
            log.warning(
                f"Invalid value '{s}' for setting '{self.name}', using default '{self.default.name}'"
            )
            log.info(f"Available options are: {', '.join(EnumType.__members__.keys())}")
            return self.default


class Settings(QObject):
    default_path = user_data_dir / "settings.json"

    language: str
    _language = Setting(
        _("Language"),
        "en",
        _("Interface language used by the plugin - requires restart!"),
    )

    auto_update: bool
    _auto_update = Setting(
        _("Enable Automatic Updates"), True, _("Check for new versions of the plugin on startup")
    )

    server_mode: ServerMode
    _server_mode = Setting(
        _("Server Management"),
        ServerMode.undefined,
        _("To generate images, the plugin connects to a ComfyUI server"),
    )

    access_token: str
    _access_token = Setting(_("Cloud Access Token"), "")

    nai_api_token: str
    _nai_api_token = Setting(
        _("NovelAI API Token"),
        "",
        _("Persistent API token (pst-xxx) for NovelAI image generation"),
    )

    nai_tokens: list
    _nai_tokens = Setting(
        _("NovelAI API Tokens"),
        [],  # [{name: str, token: str, tier: str, anlas: int}, ...]
        _("List of NovelAI API tokens for multi-account management"),
    )

    nai_active_token_index: int
    _nai_active_token_index = Setting(
        _("Active NovelAI Token Index"),
        0,
        _("Index of the currently active NovelAI API token"),
    )

    # -- NAI basic parameters (matches NAI web main panel) --

    nai_model: str
    _nai_model = Setting(
        _("Default Model"),
        "nai-diffusion-4-5-curated",
        _("NovelAI model to use for generation"),
    )

    nai_steps: int
    _nai_steps = Setting(
        _("Steps"),
        28,
        _("Number of sampling steps"),
    )

    nai_cfg_scale: float
    _nai_cfg_scale = Setting(
        _("Prompt Guidance"),
        5.0,
        _("CFG scale — how strongly the image follows the prompt"),
    )

    nai_variety_boost: bool
    _nai_variety_boost = Setting(
        _("Variety+"),
        False,
        _("Skip CFG above a sigma threshold for more diverse outputs"),
    )

    nai_variety_boost_sigma: float
    _nai_variety_boost_sigma = Setting(
        _("Variety+ Sigma"),
        19.0,
        _("Sigma threshold for Variety+ (default 19.0)"),
    )

    nai_sampler: str
    _nai_sampler = Setting(
        _("Sampler"),
        "k_euler",
        _("Sampling method for image generation"),
    )

    # -- NAI advanced parameters (matches NAI web Advanced Settings) --

    nai_cfg_rescale: float
    _nai_cfg_rescale = Setting(
        _("Prompt Guidance Rescale"),
        0.0,
        _("Rescale guidance to reduce artifacts (0.0 – 1.0)"),
    )

    nai_noise_schedule: str
    _nai_noise_schedule = Setting(
        _("Noise Schedule"),
        "native",
        _("Noise schedule used during sampling"),
    )

    # -- NAI plugin-specific settings (not on NAI web) --

    nai_uc_preset: int
    _nai_uc_preset = Setting(
        _("UC Preset"),
        0,
        _("Undesired Content preset (0=Heavy, 1=Light, 2=None)"),
    )

    nai_quality_toggle: bool
    _nai_quality_toggle = Setting(
        _("Quality Tags"),
        True,
        _("Automatically prepend quality tags to the prompt (Curated models)"),
    )

    server_path: str
    _server_path = Setting(
        _("Server Path"),
        str(user_data_dir / "server"),
        _(
            "Directory where ComfyUI will be installed. At least {size} GB of free disk space is required for a minimal installation."
        ).format(size=16),
    )

    server_url: str
    _server_url = Setting(
        _("Server URL"),
        "127.0.0.1:8188",
        _("URL used to connect to a running ComfyUI server. Default is 127.0.0.1:8188 (local)."),
    )

    server_backend: ServerBackend
    _server_backend = Setting(_("Server Backend"), ServerBackend.default())

    server_arguments: str
    _server_arguments = Setting(
        _("Server Arguments"), "", _("Additional command line arguments passed to the server")
    )

    server_authorization: str
    _server_authorization = Setting("ComfyUI Authorization Token", "")

    check_server_resources: bool
    _check_server_resources = Setting("Refuse connection if nodes or models are missing", True)

    selection_feather: int
    _selection_feather = Setting(
        _("Selection Feather"),
        10,
        _("The border is expanded and blurred by a fraction of selection size"),
    )

    selection_min_transition: int
    _selection_min_transition = Setting(
        "Selection minimum feather", 32, "Minimum smooth grow (feathering) in pixels for denoising"
    )

    selection_grow_offset: int
    _selection_grow_offset = Setting(
        "Selection Grow Offset",
        4,
        "Apply binary grow/dilation in pixels to denoise mask before smooth grow (feathering)",
    )

    selection_blend: int
    _selection_blend = Setting(
        _("Selection Blend"), 25, _("Transition area for alpha blending the result image")
    )

    selection_padding: int
    _selection_padding = Setting(
        _("Selection Padding"), 6, _("Minimum additional padding around the selection area")
    )

    color_match: bool
    _color_match = Setting(
        _("Color Match"),
        True,
        _("Match peripheral colors and brightness with existing content. Requires a selection."),
    )

    nsfw_filter: float
    _nsfw_filter = Setting(
        _("NSFW Filter"), 0.0, _("Attempt to filter out images with explicit content")
    )

    new_seed_after_apply: bool
    _new_seed_after_apply = Setting(
        _("Live: New Seed after Apply"),
        False,
        _("Pick a new seed after copying the result to the canvas in Live mode"),
    )

    prompt_translation: str
    _prompt_translation = Setting(
        _("Prompt Translation"),
        "",
        _("Translate text prompts from the selected language to English"),
    )

    save_image_metadata: bool
    _save_image_metadata = Setting(
        _("Save Image Metadata"),
        False,
        _("When saving generated images from thumbnails, include metadata in the PNG"),
    )

    save_image_format: ImageFileFormat
    _save_image_format = Setting(
        _("Save Image Format"),
        ImageFileFormat.png_small,
        _("File format for saved images from thumbnails."),
    )

    save_image_quality_webp: int
    _save_image_quality_webp = Setting(
        "Save Image Quality (WebP)",
        80,
        "Quality for WebP encoding (0-100)",
    )

    save_image_quality_jpeg: int
    _save_image_quality_jpeg = Setting(
        "Save Image Quality (JPEG)",
        85,
        "Quality for JPEG encoding (0-100)",
    )

    save_image_file_name_format: str
    _save_image_file_name_format = Setting(
        _("Save Image File Name Template"),
        "{document_name}-generated-{job_timestamp}-{job_index}-{prompt}",
        "Template for naming saved images (without extension). Available keys: {keys}.".format(
            keys="{document_name}, {job_timestamp}, {current_timestamp}, {job_index}, {prompt}"
        ),
    )

    confirm_discard_image: bool
    _confirm_discard_image = Setting("Ask for confirmation when discarding images", True)

    prompt_line_count: int
    _prompt_line_count = Setting(
        _("Prompt Line Count"), 2, _("Size of the text editor for image descriptions")
    )

    prompt_line_count_live: int
    _prompt_line_count_live = Setting("Prompt Line Count (Live)", 2)

    show_negative_prompt: bool
    _show_negative_prompt = Setting(
        _("Negative Prompt"), False, _("Show text editor to describe things to avoid")
    )

    generation_finished_action: GenerationFinishedAction
    _generation_finished_action = Setting(
        _("Finished Generation"),
        GenerationFinishedAction.preview,
        _("Action to take when an image generation job finishes"),
    )

    show_steps: bool
    _show_steps = Setting(
        _("Show Steps"), False, _("Display the number of steps to be evaluated in the weights box.")
    )

    tag_files: list[str]
    _tag_files = Setting(
        _("Tag Auto-Completion"),
        [],
        _("Enable text completion for tags from the selected files"),
    )

    apply_behavior: ApplyBehavior
    _apply_behavior = Setting(
        _("Apply Behavior"),
        ApplyBehavior.layer,
        _("Choose how result images are applied to the canvas (generation workspaces)"),
    )

    apply_region_behavior: ApplyRegionBehavior
    _apply_region_behavior = Setting("Apply Region Behavior", ApplyRegionBehavior.layer_group)

    apply_behavior_live: ApplyBehavior
    _apply_behavior_live = Setting(
        _("Apply Behavior (Live)"),
        ApplyBehavior.replace,
        _("Choose how result images are applied to the canvas in Live mode"),
    )

    apply_region_behavior_live: ApplyRegionBehavior
    _apply_region_behavior_live = Setting(
        "Apply Region Behavior (Live)", ApplyRegionBehavior.replace
    )

    show_builtin_styles: bool
    _show_builtin_styles = Setting(_("Show pre-installed styles"), True)

    recent_styles_count: int
    _recent_styles_count = Setting(
        _("Recent Styles"),
        4,
        _("Number of most recently used styles to show at the top of the style list"),
    )

    recent_styles: list[str]
    _recent_styles = Setting(
        "Recent Styles",
        [
            "built-in/edit-flux2.json",
            "built-in/anime-illustrious.json",
            "built-in/cinematic-photo-zimage.json",
            "built-in/digital-artwork-xl.json",
        ],
    )

    history_size: int
    _history_size = Setting(
        _("Active History Size"),
        1000,
        _("Main memory (RAM) used for the history of generated images"),
    )

    history_storage: int
    _history_storage = Setting(
        _("Stored History Size"),
        20,
        _("Memory used to store generated images in .kra files on disk"),
    )

    history_format: ImageFileFormat
    _history_format = Setting(
        _("History Format"),
        ImageFileFormat.webp,
        _("File format for saving generated images in history"),
    )

    multi_threading: bool
    _multi_threading = Setting(
        _("Multi-Threading"),
        True,
        _("Perform certain plugin operations in background threads"),
    )

    performance_preset: PerformancePreset
    _performance_preset = Setting(
        _("Performance Preset"),
        PerformancePreset.auto,
        _("Configures performance settings to match available hardware."),
    )

    batch_size: int
    _batch_size = Setting(
        _("Maximum Batch Size"),
        4,
        _("Increase efficiency by generating multiple images at once"),
    )

    resolution_multiplier: float
    _resolution_multiplier = Setting(
        _("Resolution Multiplier"),
        1.0,
        _(
            "Scaling factor for generation. Values below 1.0 improve performance for high resolution canvas."
        ),
    )

    max_pixel_count: int
    _max_pixel_count = Setting(
        _("Maximum Pixel Count"),
        6,
        _("Maximum resolution to generate images at, in megapixels (FullHD ~ 2MP, 4k ~ 8MP)."),
    )

    dynamic_caching: bool
    _dynamic_caching = Setting(
        _("Dynamic Caching"),
        False,
        _("Re-use outputs of previous steps (First Block Cache) to speed up generation."),
    )

    tiled_vae: bool
    _tiled_vae = Setting(
        _("Tiled VAE"),
        False,
        _("Conserve memory by processing output images in smaller tiles."),
    )

    _performance_presets: ClassVar[dict[PerformancePreset, PerformancePresetSettings]] = {
        PerformancePreset.cpu: PerformancePresetSettings(
            batch_size=1,
            resolution_multiplier=1.0,
            max_pixel_count=2,
        ),
        PerformancePreset.low: PerformancePresetSettings(
            batch_size=2,
            resolution_multiplier=1.0,
            max_pixel_count=2,
            tiled_vae=True,
        ),
        PerformancePreset.medium: PerformancePresetSettings(
            batch_size=4,
            resolution_multiplier=1.0,
            max_pixel_count=6,
        ),
        PerformancePreset.high: PerformancePresetSettings(
            batch_size=6,
            resolution_multiplier=1.0,
            max_pixel_count=8,
        ),
        PerformancePreset.cloud: PerformancePresetSettings(
            batch_size=8,
            resolution_multiplier=1.0,
            max_pixel_count=6,
        ),
    }

    debug_dump_workflow: bool
    _debug_dump_workflow = Setting(
        _("Dump Workflow"),
        False,
        _("Write latest ComfyUI prompt to the log folder for test & debug"),
    )

    document_defaults: dict[str, Any]
    _document_defaults = Setting(_("Document Defaults"), {}, _("Recently used document settings"))

    last_news: str
    _last_news = Setting("Last seen news digest", "")

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_DIFFUSION_DEBUG_IMAGE")

    changed = pyqtSignal(str, object)

    _values: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.restore(init=True)

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            if self._values[name] != value:
                self._values[name] = value
                if name != "document_defaults":
                    self.changed.emit(name, value)
                if name == "performance_preset":
                    self.apply_performance_preset(value)
        else:
            object.__setattr__(self, name, value)

    def restore(self, init=False):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }
        if not init:
            self.server_mode = ServerMode.managed

    def save(self, path: Path | None = None):
        path = self.default_path or path
        with open(path, "w") as file:
            file.write(json.dumps(self._values, default=encode_json, indent=4))

    def load(self, path: Path | None = None):
        path = self.default_path or path
        self._migrate_legacy_settings(path)
        if not path.exists():
            self.save()  # create new file with defaults
            return

        log.info(f"Loading settings from {path}")
        try:
            contents = read_json_with_comments(path)
            for k, v in contents.items():
                setting: Setting | None = getattr(Settings, f"_{k}", None)
                if setting is not None:
                    if isinstance(setting.default, Enum):
                        self._values[k] = setting.str_to_enum(v)
                    elif isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        log.error(f"{path}: {v} is not a valid value for '{k}'")
                        self._values[k] = setting.default
        except Exception as e:
            log.error(f"Failed to load settings: {e}")

        # Migrate legacy single token to multi-token list
        self._migrate_nai_tokens()

    def apply_performance_preset(self, preset: PerformancePreset):
        if preset not in [PerformancePreset.custom, PerformancePreset.auto]:
            for k, v in self._performance_presets[preset]._asdict().items():
                self._values[k] = v

    def __iter__(self):
        return iter(self._values.items())

    def get_active_nai_token(self) -> str:
        """Return the currently active NovelAI API token string.
        
        If nai_tokens list is populated, use nai_active_token_index to pick.
        Otherwise fall back to legacy nai_api_token string.
        """
        tokens = self._values.get("nai_tokens", [])
        if tokens:
            idx = self._values.get("nai_active_token_index", 0)
            idx = max(0, min(idx, len(tokens) - 1))
            return tokens[idx].get("token", "")
        return self._values.get("nai_api_token", "")

    def set_active_nai_token_index(self, index: int):
        """Switch active token and sync nai_api_token for backward compat."""
        tokens = self._values.get("nai_tokens", [])
        if 0 <= index < len(tokens):
            self._values["nai_active_token_index"] = index
            self._values["nai_api_token"] = tokens[index].get("token", "")
            self.changed.emit("nai_active_token_index", index)
            self.changed.emit("nai_api_token", self._values["nai_api_token"])

    def _migrate_nai_tokens(self):
        """Migrate legacy single nai_api_token to nai_tokens list if needed."""
        tokens = self._values.get("nai_tokens", [])
        old_token = self._values.get("nai_api_token", "")
        if not tokens and old_token:
            self._values["nai_tokens"] = [
                {"name": "Default", "token": old_token, "tier": "", "anlas": 0}
            ]
            self._values["nai_active_token_index"] = 0
            log.info("Migrated legacy nai_api_token to nai_tokens list")

    def _migrate_legacy_settings(self, path: Path):
        if path == self.default_path:
            legacy_path = Path(__file__).parent / "settings.json"
            if legacy_path.exists() and not path.exists():
                try:
                    legacy_path.rename(path)
                    log.info(f"Migrated settings from {legacy_path} to {path}")
                except Exception as e:
                    log.warning(f"Failed to migrate settings from {legacy_path} to {path}: {e}")


settings = Settings()
