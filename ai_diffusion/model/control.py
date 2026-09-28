from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from PyQt5.QtCore import QMetaObject, QObject, Qt, QTimer, QUuid, pyqtSignal

from .. import util
from ..backend import resources
from ..backend.api import ControlInput
from ..backend.nai_workflow import NaiModel, nai_auto_resolution
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
    enabled = Property(True, persist=True)
    group_id = Property(QUuid(), persist=True)
    group_enabled = Property(True, persist=True)
    group_expanded = Property(True, persist=True)
    layer_id = Property(QUuid(), persist=True, setter="set_layer_id")
    preset_value = Property(2, persist=True, setter="set_preset_value")
    strength = Property(50, persist=True)
    start = Property(0.0, persist=True)
    end = Property(1.0, persist=True)
    # Secondary parameter for NAI modes, in percent (0-100): vibe "information
    # extracted" (default 70) or precise-reference "fidelity" (default 100).
    param2 = Property(100, persist=True)
    use_custom_strength = Property(False, persist=True, setter="set_use_custom_strength")
    is_supported = Property(True)
    is_pose_vector = Property(False)
    can_generate = Property(True)
    has_range = Property(True)
    has_active_job = Property(False)
    error_text = Property("")

    mode_changed = pyqtSignal(ControlMode)
    enabled_changed = pyqtSignal(bool)
    group_id_changed = pyqtSignal(QUuid)
    group_enabled_changed = pyqtSignal(bool)
    group_expanded_changed = pyqtSignal(bool)
    layer_id_changed = pyqtSignal(QUuid)
    preset_value_changed = pyqtSignal(int)
    strength_changed = pyqtSignal(int)
    start_changed = pyqtSignal(float)
    end_changed = pyqtSignal(float)
    param2_changed = pyqtSignal(int)
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
        self._selection_bounds_connection: QMetaObject.Connection | None = None
        self._selection_target_timer = QTimer(self)
        self._selection_target_timer.setInterval(100)
        self._selection_target_timer.setSingleShot(True)
        self._selection_target_timer.timeout.connect(self._update_selection_target)
        self.layer_id = layer_id
        self.mode = mode
        self._update_is_supported()

        self.mode_changed.connect(self._update_is_supported)
        self.enabled_changed.connect(self._update_enabled)
        self.group_enabled_changed.connect(self._update_enabled)
        model.nai_cropped_inpaint_changed.connect(self._sync_selection_bounds_connection)
        model.style_changed.connect(self._update_is_supported)
        model.edit_mode_changed.connect(self._update_is_supported)
        root.connection.state_changed.connect(self._update_is_supported)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        self._sync_selection_bounds_connection()
        model.document_changed.connect(self._update_document)
        model.jobs.job_finished.connect(self._update_active_job)

    @property
    def model(self):
        return self._model

    @property
    def is_active(self):
        return self.enabled and self.group_enabled

    @property
    def is_whole_canvas(self):
        """NAI modes may target the whole canvas instead of a single layer."""
        return self.layer_id.isNull()

    @property
    def is_selection(self):
        return self.layer_id == nai_selection_layer_id

    @property
    def reference(self):
        return self._model.nai_references.find(self.layer_id)

    @property
    def layer(self):
        if self.layer_id.isNull() or self.is_selection or self.reference is not None:
            return None
        layer = self._model.layers.find(self.layer_id)
        return layer

    def set_mode(self, mode: ControlMode):
        if mode != self.mode:
            if not mode.is_nai and self.reference is not None:
                self.layer_id = self._model.layers.active.id
            if mode is not ControlMode.nai_base and self.is_selection:
                self.layer_id = QUuid()
            self._mode = mode
            self._sync_selection_bounds_connection()
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
            self.modified.emit(self, "mode")

    def set_layer_id(self, layer_id: QUuid):
        if layer_id != self.layer_id:
            self._layer_id = layer_id
            self._sync_selection_bounds_connection()
            self.layer_id_changed.emit(layer_id)
            self.modified.emit(self, "layer_id")
            if self.mode is ControlMode.nai_base:
                self._reset_target_resolution()

    @property
    def target_extent(self):
        """There is one output resolution and it is the docker's.

        An img2img layer used to keep a second, private one: no widget showed
        it and no edit could reach it, so the number on screen and the number
        actually sent drifted apart. Now the layer only *seeds* the docker
        value (see _reset_target_resolution) and reads it back here."""
        return self._model.nai_target_extent

    @property
    def source_extent(self):
        """Extent of the pixels this layer sends - what the target follows."""
        return self._source_extent()

    def _source_extent(self):
        if reference := self.reference:
            return reference.extent
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
        if self.is_active and not self._model.nai_cropped_inpaint:
            self._model.set_nai_target_extent(nai_auto_resolution(self._source_extent()))

    def _update_enabled(self):
        self._sync_selection_bounds_connection()
        if self.mode is ControlMode.nai_base and not self._model.nai_cropped_inpaint:
            self._model.reset_nai_target_resolution()

    def _update_selection_target(self):
        bounds = self._model.document.selection_bounds
        if (
            self.is_active
            and self.mode is ControlMode.nai_base
            and self.is_selection
            and not self._model.nai_cropped_inpaint
            and bounds
        ):
            self._model.set_nai_target_extent(nai_auto_resolution(bounds.extent))

    def _schedule_selection_target_update(self):
        self._selection_target_timer.start()

    def _sync_selection_bounds_connection(self):
        if self._selection_bounds_connection is not None:
            QObject.disconnect(self._selection_bounds_connection)
            self._selection_bounds_connection = None
        if (
            self.is_active
            and self.mode is ControlMode.nai_base
            and self.is_selection
            and not self._model.nai_cropped_inpaint
        ):
            self._selection_bounds_connection = (
                self._model.document.selection_bounds_changed.connect(
                    self._schedule_selection_target_update
                )
            )
        else:
            self._selection_target_timer.stop()

    def _update_document(self, document: Document):
        self._sync_selection_bounds_connection()
        self._update_selection_target()

    @property
    def selection_bounds(self):
        if not self.is_selection:
            return None
        if bounds := self._model.document.selection_bounds:
            return bounds
        _, bounds = self._model.document.create_mask_from_selection(SelectionModifiers(multiple=1))
        return bounds

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
        reference = self.reference
        if (
            layer is None
            and reference is None
            and not self.is_whole_canvas
            and not self.is_selection
        ):
            raise PluginError("图像来源不存在，请迁移本机素材库或重新选择素材/图层")
        layer_name = (
            reference.name
            if reference
            else (
                _("Selection") if self.is_selection else layer.name if layer else _("Whole canvas")
            )
        )
        if not self.is_supported:
            raise PluginError(f"Can't use '{layer_name}' as control layer: {self.error_text}")

        if self.mode.is_nai:
            # NAI reference/base images: full-resolution, no CLIP-Vision downscale,
            # no line/stencil preprocessing. Whole-canvas (null layer_id) uses the
            # flattened document projection minus control/preview layers.
            doc_bounds = Bounds(0, 0, *self._model.document.extent)
            if reference is not None:
                image = self._model.nai_references.image(reference.id)
            elif self.is_selection:
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
                elif is_nai_arch and self.mode.is_nai:
                    checkpoints = self._model.style.checkpoints
                    checkpoint = checkpoints[0] if checkpoints else ""
                    try:
                        nai_model = NaiModel(checkpoint)
                    except ValueError:
                        nai_model = None
                    if nai_model is not None and nai_model.is_v5:
                        if self.mode is ControlMode.nai_vibe:
                            self.error_text = _("Vibe Transfer is not available for NAI V5 yet")
                            is_supported = False
                        elif self.mode.is_nai_precise:
                            self.error_text = _("Precise Reference is not available for NAI V5 yet")
                            is_supported = False
                if is_supported and self._index >= client.features.max_control_layers:
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
    # Anything that can change what the list *means*: added, removed, or a
    # layer switching mode. Widgets that gray themselves out for an img2img
    # layer need all three, and used to see none of them.
    changed = pyqtSignal()

    _model: model.DocumentModel
    _layers: list[ControlLayer]
    _last_mode = ControlMode.scribble

    def __init__(self, model: model.DocumentModel):
        super().__init__()
        self._model = model
        self._layers = []
        self._control_connections: dict[ControlLayer, list[QMetaObject.Connection]] = {}
        self._syncing_group = False
        self._model.layers.removed.connect(self._remove_layer)
        self._model.nai_references.removing.connect(self._remove_reference)

    def add(self):
        layer = self._model.layers.active
        if layer.type.is_filter and layer.parent_layer and not layer.parent_layer.is_root:
            layer = layer.parent_layer
        if not layer.type.is_image:
            layer = next(iter(self._model.layers.images), None)
        if layer is None:  # shouldn't be possible, Krita doesn't allow removing all non-mask layers
            log.warning("Trying to add control layer, but document has no suitable layer")
            return
        layer_id = layer.id
        if self._model.arch is Arch.nai:
            # Always 图生图-选区, not the last mode used: it is where nearly every
            # NAI job in this fork starts, and the selection is the only source
            # whose size the docker's target resolution follows.
            mode = ControlMode.nai_base
            layer_id = nai_selection_layer_id
        elif self._model.arch.is_edit:
            mode = ControlMode.reference
        elif self._last_mode.is_nai:
            mode = ControlMode.scribble
        else:
            mode = self._last_mode
        index = sum(layer.is_active for layer in self._layers)
        control = ControlLayer(self._model, mode, layer_id, index)
        self._connect_control(control)
        self._layers.append(control)
        self.added.emit(control)
        self.changed.emit()

    def group_members(self, control: ControlLayer):
        if control.group_id.isNull():
            return [control]
        return [layer for layer in self._layers if layer.group_id == control.group_id]

    def add_related(self, control: ControlLayer):
        if control not in self._layers or control.mode is ControlMode.nai_base:
            return None
        if control.group_id.isNull():
            control.group_id = QUuid.createUuid()
        members = self.group_members(control)
        index = self._layers.index(members[-1]) + 1
        related = ControlLayer(self._model, control.mode, control.layer_id, index)
        for name in (
            "group_id",
            "group_enabled",
            "use_custom_strength",
            "preset_value",
            "strength",
            "start",
            "end",
            "param2",
        ):
            setattr(related, name, getattr(control, name))
        control.group_expanded = True
        self._connect_control(related)
        self._layers.insert(index, related)
        self.added.emit(related)
        self._update_enabled()
        return related

    def _connect_control(self, control: ControlLayer):
        self._control_connections[control] = [
            control.mode_changed.connect(self._update_last_mode),
            control.mode_changed.connect(lambda: self._sync_group(control, "mode")),
            control.enabled_changed.connect(self._update_enabled),
            control.group_id_changed.connect(self.changed),
            control.group_enabled_changed.connect(
                lambda: self._sync_group(control, "group_enabled")
            ),
            control.group_expanded_changed.connect(
                lambda: self._sync_group(control, "group_expanded")
            ),
        ]

    def _sync_group(self, control: ControlLayer, name: str):
        if self._syncing_group:
            return
        self._syncing_group = True
        try:
            for member in self.group_members(control):
                if member is not control:
                    setattr(member, name, getattr(control, name))
        finally:
            self._syncing_group = False
        self._update_enabled()

    def emplace(self):
        self.add()
        return self[-1]

    def remove(self, control: ControlLayer):
        was_base = control.is_active and control.mode is ControlMode.nai_base
        self._layers.remove(control)
        for connection in self._control_connections.pop(control):
            QObject.disconnect(connection)
        self.removed.emit(control)
        self._update_enabled()
        if was_base and not self._model.nai_cropped_inpaint:
            # The img2img source is gone; the target goes back to the canvas.
            self._model.reset_nai_target_resolution()

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        for layer in (c for c in self._layers if c.is_active and not c.is_supported):
            log.warning(f"Trying to use control layer {layer.mode.name}: {layer.error_text}")
        return [c.to_api(bounds, time) for c in self._layers if c.is_active and c.is_supported]

    def _update_enabled(self):
        index = 0
        for control in self._layers:
            control.index = index
            if control.is_active:
                index += 1
        self.changed.emit()

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _remove_layer(self, layer: Layer):
        for control in list(self._layers):
            if control.layer_id == layer.id:
                self.remove(control)

    def _remove_reference(self, reference_id: QUuid):
        for control in list(self._layers):
            if control.layer_id == reference_id:
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
