from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from PyQt5.QtCore import QMetaObject, QObject, Qt, QUuid, pyqtSignal

from .. import util
from ..backend import resources
from ..backend.api import ControlInput
from ..backend.nai_workflow import nai_auto_resolution, nai_edit_resolution
from ..backend.resources import Arch, ControlMode, ResourceKind, resource_id
from ..document import Document, SelectionModifiers
from ..image import Bounds, Extent, Image
from ..layer import Layer, LayerType
from ..localization import translate as _
from ..util import PluginError
from ..util import client_logger as log
from . import jobs
from .control_utils import nai_selection_layer_id
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from . import model


class ControlLayer(QObject, ObservableProperties):
    max_preset_value = 4
    strength_multiplier = 50
    clip_vision_extent = Extent(224, 224)

    mode = Property(ControlMode.reference, persist=True, setter="set_mode")
    layer_id = Property(QUuid(), persist=True, setter="set_layer_id")
    preset_value = Property(2, persist=True, setter="set_preset_value")
    strength = Property(50, persist=True)
    start = Property(0.0, persist=True)
    end = Property(1.0, persist=True)
    # Secondary parameter for NAI modes, in percent (0-100): vibe "information
    # extracted" (default 70) or precise-reference "fidelity" (default 100).
    param2 = Property(100, persist=True)
    target_width = Property(1024, persist=True, setter="set_target_width")
    target_height = Property(1024, persist=True, setter="set_target_height")
    use_custom_strength = Property(False, persist=True, setter="set_use_custom_strength")
    is_supported = Property(True)
    is_pose_vector = Property(False)
    can_generate = Property(True)
    has_range = Property(True)
    has_active_job = Property(False)
    error_text = Property("")

    mode_changed = pyqtSignal(ControlMode)
    layer_id_changed = pyqtSignal(QUuid)
    preset_value_changed = pyqtSignal(int)
    strength_changed = pyqtSignal(int)
    start_changed = pyqtSignal(float)
    end_changed = pyqtSignal(float)
    param2_changed = pyqtSignal(int)
    target_width_changed = pyqtSignal(int)
    target_height_changed = pyqtSignal(int)
    use_custom_strength_changed = pyqtSignal(bool)
    is_supported_changed = pyqtSignal(bool)
    is_pose_vector_changed = pyqtSignal(bool)
    can_generate_changed = pyqtSignal(bool)
    has_active_job_changed = pyqtSignal(bool)
    has_range_changed = pyqtSignal(bool)
    error_text_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: model.DocumentModel, mode: ControlMode, layer_id: QUuid, index: int):
        from .root import root

        super().__init__()
        self._model = model
        self._index = index
        self._generate_job: jobs.Job | None = None
        self.layer_id = layer_id
        self.mode = mode
        self._update_is_supported()

        self.mode_changed.connect(self._update_is_supported)
        model.style_changed.connect(self._update_is_supported)
        model.edit_mode_changed.connect(self._update_is_supported)
        root.connection.state_changed.connect(self._update_is_supported)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        self._selection_bounds_connection: QMetaObject.Connection = (
            model.document.selection_bounds_changed.connect(self._update_selection_target)
        )
        model.document_changed.connect(self._update_document)
        model.jobs.job_finished.connect(self._update_active_job)

    @property
    def is_whole_canvas(self):
        """NAI modes may target the whole canvas instead of a single layer."""
        return self.layer_id.isNull()

    @property
    def is_selection(self):
        return self.layer_id == nai_selection_layer_id

    @property
    def layer(self):
        if self.layer_id.isNull() or self.is_selection:
            return None
        layer = self._model.layers.updated().find(self.layer_id)
        assert layer is not None, "Control layer has been deleted"
        return layer

    def set_mode(self, mode: ControlMode):
        if mode != self.mode:
            if mode is not ControlMode.nai_base and self.is_selection:
                self.layer_id = QUuid()
            self._mode = mode
            self.mode_changed.emit(mode)
            self._update_is_pose_vector()
            if mode.is_nai:
                # nai_base: param2 = noise (default 0); nai_vibe: information
                # extracted (default 0.7); precise: fidelity (default 1.0).
                if mode is ControlMode.nai_base:
                    self.param2 = 0
                    self._reset_target_resolution()
                elif mode is ControlMode.nai_vibe:
                    self.param2 = 70
                else:
                    self.param2 = 100
            if not self.use_custom_strength:
                self._set_values_from_preset()

    def set_layer_id(self, layer_id: QUuid):
        if layer_id != self.layer_id:
            self._layer_id = layer_id
            self.layer_id_changed.emit(layer_id)
            self.modified.emit(self, "layer_id")
            if self.mode is ControlMode.nai_base:
                self._reset_target_resolution()

    @property
    def target_extent(self):
        return Extent(self.target_width, self.target_height)

    def _set_target_extent(self, extent: Extent):
        target = nai_edit_resolution(extent)
        for name, value in (("target_width", target.width), ("target_height", target.height)):
            if value != getattr(self, name):
                setattr(self, f"_{name}", value)
                getattr(self, f"{name}_changed").emit(value)
                self.modified.emit(self, name)

    def set_target_width(self, value: int):
        self._set_target_extent(Extent(value, self.target_height))

    def set_target_height(self, value: int):
        self._set_target_extent(Extent(self.target_width, value))

    def set_target_extent(self, extent: Extent):
        self._set_target_extent(extent)

    def _source_extent(self):
        if self.is_selection:
            if bounds := self.selection_bounds:
                return bounds.extent
            return self._model.document.extent
        layer = self.layer
        if layer is None:
            return self._model.document.extent
        bounds = layer.full_bounds
        return bounds.extent if not bounds.is_zero else self._model.document.extent

    def _reset_target_resolution(self):
        self.set_target_extent(nai_auto_resolution(self._source_extent()))

    def _update_selection_target(self):
        bounds = self._model.document.selection_bounds
        if self.mode is ControlMode.nai_base and self.is_selection and bounds:
            self.set_target_extent(nai_auto_resolution(bounds.extent))

    def _update_document(self, document: Document):
        QObject.disconnect(self._selection_bounds_connection)
        self._selection_bounds_connection = document.selection_bounds_changed.connect(
            self._update_selection_target
        )
        self._update_selection_target()

    @property
    def selection_bounds(self):
        if not self.is_selection:
            return None
        if bounds := self._model.document.selection_bounds:
            return bounds
        _, bounds = self._model.document.create_mask_from_selection(SelectionModifiers(multiple=1))
        return bounds

    def adapt_target_resolution(self):
        source = self._source_extent()
        if source.width >= source.height:
            width = self.target_width
            height = round(width * source.height / max(source.width, 1))
        else:
            height = self.target_height
            width = round(height * source.width / max(source.height, 1))
        self.set_target_extent(Extent(width, height))

    def set_preset_value(self, value: int):
        if value != self.preset_value:
            self._preset_value = value
            self.preset_value_changed.emit(value)
            self._set_values_from_preset()

    def _set_values_from_preset(self):
        if self.mode.is_nai:
            # No entries in presets/control.json for NAI modes. Launcher defaults:
            # img2img strength 0.7, vibe strength 0.6, precise-reference 1.0.
            if self.mode is ControlMode.nai_base:
                default = 0.7
            elif self.mode is ControlMode.nai_vibe:
                default = 0.6
            else:
                default = 1.0
            self.strength = int(default * self.strength_multiplier)
            self.start, self.end = 0.0, 1.0
            return
        params = ControlPresets.instance().interpolate(
            self.mode, self._model.arch, self.preset_value / self.max_preset_value
        )
        self.strength = int(params.strength * self.strength_multiplier)
        self.start, self.end = params.range

    def set_use_custom_strength(self, value: bool):
        if value != self.use_custom_strength:
            self._use_custom_strength = value
            self.use_custom_strength_changed.emit(value)
            if not value:
                self._set_values_from_preset()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        self._update_is_supported()

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        layer = self.layer
        layer_name = (
            _("Selection") if self.is_selection else layer.name if layer else _("Whole canvas")
        )
        if not self.is_supported:
            raise PluginError(f"Can't use '{layer_name}' as control layer: {self.error_text}")

        if self.mode.is_nai:
            # NAI reference/base images: full-resolution, no CLIP-Vision downscale,
            # no line/stencil preprocessing. Whole-canvas (null layer_id) uses the
            # flattened document projection minus control/preview layers.
            doc_bounds = Bounds(0, 0, *self._model.document.extent)
            if self.is_selection:
                selection_bounds = self.selection_bounds
                if selection_bounds is None:
                    raise PluginError(_("There is no active selection for img2img"))
                image = self._model._get_current_image(selection_bounds)
            elif layer is None:
                image = self._model._get_current_image(doc_bounds)
            elif self.mode is ControlMode.nai_base:
                # A specific layer as img2img base = the FULL original image (not
                # clamped to the canvas), stretched to the request resolution later
                # — same as importing an image in the launcher. "Whole canvas"
                # (layer is None above) captures the canvas window instead.
                base_bounds = layer.full_bounds
                if base_bounds.is_zero:
                    base_bounds = doc_bounds
                image = layer.get_pixels(base_bounds, time)
            else:
                # Vibe / precise reference: STRICTLY the layer's original pixels at
                # full resolution — full_bounds is NOT clamped to the canvas, so
                # content outside the canvas is preserved, nothing is cropped or
                # rescaled here.
                ref_bounds = layer.full_bounds
                if ref_bounds.is_zero:
                    ref_bounds = doc_bounds
                image = layer.get_pixels(ref_bounds, time)
            strength = min(self.strength / self.strength_multiplier, 1.0)
            param2 = min(max(self.param2 / 100.0, 0.0), 1.0)
            target = self.target_extent if self.mode is ControlMode.nai_base else None
            return ControlInput(self.mode, image, strength, (0.0, 1.0), param2, target)

        assert layer is not None, "Control layer has been deleted"
        extent = bounds.extent if bounds else self._model.document.extent
        if self.mode.is_ip_adapter and not layer.bounds.is_zero:
            bounds = None  # ignore mask bounds, use layer bounds

        image = layer.get_pixels(bounds, time)

        if self.mode.is_lines or self.mode is ControlMode.stencil:
            image.make_opaque(background=Qt.GlobalColor.white)

        if self.mode.is_ip_adapter:
            if self._model.arch.supports_edit:
                if image.extent.height > extent.height:
                    w = (image.extent.width * extent.height) // image.extent.height
                    image = Image.scale(image, Extent(w, extent.height))
            else:
                image = Image.scale(image, self.clip_vision_extent)

        strength = self.strength / self.strength_multiplier
        return ControlInput(self.mode, image, strength, (self.start, self.end))

    def generate(self):
        self._generate_job = self._model.generate_control_layer(self)
        self.has_active_job = True

    def _update_is_supported(self):
        from .root import root

        is_supported = True
        if client := root.connection.client_if_connected:
            models = client.models.for_arch(self._model.arch)

            is_nai_arch = self._model.arch is Arch.nai
            if self.mode.is_nai or is_nai_arch:
                if self.mode.is_nai and not is_nai_arch:
                    self.error_text = _("Only available with the NovelAI backend")
                    is_supported = False
                elif is_nai_arch and not self.mode.is_nai:
                    self.error_text = _("Not supported for") + " NovelAI"
                    is_supported = False
                elif self._index >= client.features.max_control_layers:
                    self.error_text = _("Too many control layers")
                    is_supported = False
                self.is_supported = is_supported
                self.can_generate = False
                return

            if self.mode.is_ip_adapter and models.arch in [Arch.illu, Arch.illu_v]:
                resid = resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
                has_clip_vision = client.models.resources.get(resid, None) is not None
                if not has_clip_vision:
                    search = resources.search_path(
                        ResourceKind.clip_vision, Arch.illu, "ip_adapter"
                    )
                    self.error_text = _("The server is missing the ClipVision model") + f" {search}"
                    is_supported = False

            if self.mode.is_ip_adapter and models.arch.supports_edit:
                is_supported = True  # Reference images are merged into the conditioning context
            elif self.mode.is_ip_adapter and models.ip_adapter.find(self.mode) is None:
                search_path = resources.search_path(ResourceKind.ip_adapter, models.arch, self.mode)
                if search_path:
                    self.error_text = (
                        _("The server is missing the IP-Adapter model") + f" {self.mode.text}"
                    )
                else:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
                if not client.features.ip_adapter:
                    self.error_text = _("IP-Adapter is not supported by this GPU")
                is_supported = False
            elif self.mode.is_control_net and models.arch.supports_edit:
                is_supported = self.mode.can_substitute_instruction(models.arch)
                if not is_supported:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
            elif self.mode.is_control_net:
                model = models.find_control(self.mode)
                self.has_range = model == models.control.find(self.mode, True)
                if model is None:
                    search_arch = Arch.illu if models.arch is Arch.illu_v else models.arch
                    search_path = (
                        resources.search_path(ResourceKind.controlnet, search_arch, self.mode)
                        or resources.search_path(ResourceKind.model_patch, search_arch, self.mode)
                        or resources.search_path(ResourceKind.lora, models.arch, self.mode)
                    )
                    if search_path:
                        self.error_text = (
                            _("The ControlNet model is not installed") + f" {search_path}"
                        )
                    else:
                        self.error_text = _("Not supported for") + f" {models.arch.value}"
                    is_supported = False

            if self._index >= client.features.max_control_layers:
                self.error_text = _("Too many control layers")
                is_supported = False

        self.is_supported = is_supported
        self.can_generate = is_supported and self.mode.has_preprocessor

    def _update_is_pose_vector(self):
        layer = self.layer
        self.is_pose_vector = (
            self.mode is ControlMode.pose and layer is not None and layer.type is LayerType.vector
        )

    def _update_active_job(self):
        from .jobs import JobState

        active = not (self._generate_job is None or self._generate_job.state is JobState.finished)
        if self.has_active_job and not active:
            self._job = None  # job done
        self.has_active_job = active


class ControlLayerList(QObject):
    """List of control layers for one document."""

    added = pyqtSignal(ControlLayer)
    removed = pyqtSignal(ControlLayer)

    _model: model.DocumentModel
    _layers: list[ControlLayer]
    _last_mode = ControlMode.scribble

    def __init__(self, model: model.DocumentModel):
        super().__init__()
        self._model = model
        self._layers = []
        self._model.layers.removed.connect(self._remove_layer)

    def add(self):
        layer = self._model.layers.active
        if layer.type.is_filter and layer.parent_layer and not layer.parent_layer.is_root:
            layer = layer.parent_layer
        if not layer.type.is_image:
            layer = next(iter(self._model.layers.images), None)
        if layer is None:  # shouldn't be possible, Krita doesn't allow removing all non-mask layers
            log.warning("Trying to add control layer, but document has no suitable layer")
            return
        if self._model.arch is Arch.nai:
            mode = self._last_mode if self._last_mode.is_nai else ControlMode.nai_vibe
        elif self._model.arch.is_edit:
            mode = ControlMode.reference
        elif self._last_mode.is_nai:
            mode = ControlMode.scribble
        else:
            mode = self._last_mode
        control = ControlLayer(self._model, mode, layer.id, len(self._layers))
        control.mode_changed.connect(self._update_last_mode)
        self._layers.append(control)
        self.added.emit(control)

    def emplace(self):
        self.add()
        return self[-1]

    def remove(self, control: ControlLayer):
        self._layers.remove(control)
        self.removed.emit(control)

        for i, c in enumerate(self._layers):
            c.index = i

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        for layer in (c for c in self._layers if not c.is_supported):
            log.warning(f"Trying to use control layer {layer.mode.name}: {layer.error_text}")
        return [c.to_api(bounds, time) for c in self._layers if c.is_supported]

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _remove_layer(self, layer: Layer):
        if control := next((c for c in self._layers if c.layer_id == layer.id), None):
            self.remove(control)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __iter__(self):
        return iter(self._layers)


class ControlParams(NamedTuple):
    strength: float
    range: tuple[float, float]

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return ControlParams(data["strength"], (data["start"], data["end"]))


class ControlPresets:
    _path: Path
    _user_path: Path
    _presets: dict[str, dict[str, list[dict[str, Any]]]]

    _instance: ControlPresets | None = None

    @classmethod
    def instance(cls) -> ControlPresets:
        if cls._instance is None:
            cls._instance = ControlPresets()
        return cls._instance

    def __init__(self):
        self._path = util.plugin_dir / "presets" / "control.json"
        self._user_path = util.user_data_dir / "presets" / "control.json"
        self._read()

    def get(self, mode: ControlMode, arch: Arch):
        default = self._presets["default"]
        versions = self._presets.get(mode.name, default)
        all = versions.get("all", None)
        presets = versions.get(arch.name, all)
        if presets is None:
            raise PluginError(f"No control strength presets found for {mode} and {arch}")
        return [ControlParams.from_dict(p) for p in presets]

    def interpolate(self, mode: ControlMode, arch: Arch, value: float):
        assert value >= 0 and value <= 1, f"Interpolate value out of range: {value}"
        presets = self.get(mode, arch)
        if len(presets) == 1 or value <= 0:
            return presets[0]
        if value == 1:
            return presets[-1]
        value = value * (len(presets) - 1)
        for i, p0 in enumerate(presets):
            if value < i + 1:
                p1 = presets[i + 1]
                t = value - i
                return ControlParams(
                    _lerp(p0.strength, p1.strength, t),
                    (_lerp(p0.range[0], p1.range[0], t), _lerp(p0.range[1], p1.range[1], t)),
                )
        assert False, f"Interpolation failed: {mode}, {arch}, value={value}, presets={presets}"

    def _read(self):
        self._presets = self._read_file(self._path)
        _validate_presets(self._path, self._presets)
        if self._user_path.exists():
            user = self._read_file(self._user_path)
            if _validate_presets(self._user_path, user):
                _recursive_update(self._presets, user)
        else:
            self._user_path.parent.mkdir(parents=True, exist_ok=True)
            self._user_path.write_text(json.dumps({}, indent=4))

    def _read_file(self, path: Path):
        try:
            return json.load(path.open("r"))
        except Exception as e:
            raise ValueError(f"Failed to read control layer presets file {path}: {e}") from e


def _validate_presets(filepath: Path, data: dict[str, Any]) -> bool:
    control_modes = ["default"] + list(ControlMode.__members__.keys())
    model_archs = list(Arch.__members__.keys())

    for mode, versions in data.items():
        if mode not in control_modes:
            log.error(
                f"Invalid control mode '{mode}' in presets file {filepath}."
                f" Valid modes are: {', '.join(control_modes)}"
            )
            return False
        if not isinstance(versions, dict):
            log.error(f"Invalid presets for mode '{mode}' in presets file {filepath}.")
            return False
        for arch, presets in versions.items():
            if arch not in model_archs:
                log.error(
                    f"Invalid Base model '{arch}' for mode '{mode}' in presets file {filepath}."
                    f" Valid versions are: {', '.join(model_archs)}"
                )
                return False
            if not isinstance(presets, list):
                log.error(
                    f"Invalid presets for '{mode}/{arch}' in presets file {filepath}."
                    f" Expected a list, got {presets}"
                )
                return False
            for p in presets:
                if not isinstance(p, dict) or not all(k in p for k in ("strength", "start", "end")):
                    log.error(
                        f"Invalid preset for '{mode}/{arch}' in presets file {filepath}."
                        f" Expected a {{strength, start, end}}, got {p}"
                    )
                    return False
    return True


control_mode_text = {
    ControlMode.reference: _("Reference"),
    ControlMode.inpaint: _("Inpaint"),
    ControlMode.style: _("Style"),
    ControlMode.composition: _("Composition"),
    ControlMode.face: _("Face"),
    ControlMode.universal: _("Universal"),
    ControlMode.scribble: _("Scribble"),
    ControlMode.line_art: _("Line Art"),
    ControlMode.soft_edge: _("Soft Edge"),
    ControlMode.canny_edge: _("Canny Edge"),
    ControlMode.depth: _("Depth"),
    ControlMode.normal: _("Normal"),
    ControlMode.pose: _("Pose"),
    ControlMode.segmentation: _("Segment"),
    ControlMode.blur: _("Unblur"),
    ControlMode.stencil: _("Stencil"),
    ControlMode.hands: _("Hands"),
    ControlMode.nai_base: "图生图",
    ControlMode.nai_vibe: "Vibe Transfer",
    ControlMode.nai_precise_character: "精准参考-角色",
    ControlMode.nai_precise_style: "精准参考-风格",
    ControlMode.nai_precise_character_style: "精准参考-角色&风格",
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _recursive_update(a: dict[str, Any], b: dict[str, Any]):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _recursive_update(a.get(k, {}), v)
        else:
            a[k] = v
    return a
