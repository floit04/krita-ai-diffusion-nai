from __future__ import annotations

from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QMetaObject, QUuid, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QResizeEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.resources import Arch, ControlMode
from ..image import Extent
from ..localization import translate as _
from ..model.control import ControlLayer, ControlLayerList
from ..model.properties import Binding, bind, bind_combo, bind_toggle
from ..model.root import root
from . import theme
from .interval_slider import IntervalSlider
from .theme import SignalBlocker

_thumbnail_extent = Extent(192, 192)


def _thumbnail_tooltip(image: QImage | None, title: str) -> str:
    """HTML tooltip with an embedded preview image (shown on combo item hover)."""
    import html

    if image is None or image.isNull():
        return html.escape(title)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    b64 = bytes(data.toBase64()).decode("ascii")
    return f"<b>{html.escape(title)}</b><br><img src='data:image/png;base64,{b64}'>"


class ControlWidget(QWidget):
    def __init__(
        self, control_list: ControlLayerList | None, control: ControlLayer, parent: QWidget
    ):
        super().__init__(parent)
        self._control_list = control_list
        self._control = control
        self._connections: list[QMetaObject.Connection | Binding] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.mode_select = QComboBox(self)
        self.mode_select.setStyleSheet(theme.flat_combo_stylesheet)
        self._update_modes()

        self.layer_select = QComboBox(self)
        self.layer_select.setMinimumContentsLength(20)
        self.layer_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self._update_layers()
        root.active_model.layers.changed.connect(self._update_layers)
        control.mode_changed.connect(self._update_layers)

        self.preset_slider = QSlider(self)
        self.preset_slider.setOrientation(Qt.Orientation.Horizontal)
        self.preset_slider.setMinimumWidth(40)
        self.preset_slider.setRange(0, control.max_preset_value)
        self.preset_slider.setValue(control.preset_value)
        self.preset_slider.setSingleStep(1)
        self.preset_slider.setPageStep(2)
        self.preset_slider.setTickInterval(2)
        self.preset_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.preset_slider.setToolTip(_("Control strength: how much the layer affects the image"))

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet(f"QLabel {{ color: {theme.yellow}; }}")
        self.error_text.setVisible(not control.is_supported)
        self._set_error(control.error_text)

        self.generate_tool_button = _create_generate_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.generate_tool_button.clicked.connect(control.generate)

        self.add_pose_tool_button = _create_add_pose_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.add_pose_tool_button.clicked.connect(self._add_pose_character)

        self.expand_button = QToolButton(self)
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.expand_button.setIcon(theme.icon("more"))
        self.expand_button.setToolTip(_("Show/hide advanced settings"))
        self.expand_button.setAutoRaise(True)
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(False)
        self.expand_button.clicked.connect(self._toggle_extended)

        bar_layout = QHBoxLayout()
        bar_layout.addWidget(self.mode_select)
        bar_layout.addWidget(self.layer_select, 3)
        bar_layout.addWidget(self.generate_tool_button)
        bar_layout.addWidget(self.add_pose_tool_button)
        bar_layout.addWidget(self.preset_slider, 1)
        bar_layout.addWidget(self.error_text, 3)
        bar_layout.addWidget(self.expand_button)
        layout.addLayout(bar_layout)

        if self._control_list is not None:
            self.remove_button = QToolButton(self)
            self.remove_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.remove_button.setIcon(theme.icon("remove"))
            self.remove_button.setToolTip(_("Remove control layer"))
            self.remove_button.setAutoRaise(True)
            self.remove_button.clicked.connect(self.remove)
            bar_layout.addWidget(self.remove_button)

        line = QFrame(self)
        line.setObjectName("LeftIndent")
        line.setStyleSheet(f"#LeftIndent {{ color: {theme.line};  }}")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setLineWidth(1)

        extended_layout = QVBoxLayout()
        extended_layout.setContentsMargins(8, 2, 4, 6)
        pad_layout = QHBoxLayout()
        pad_layout.setContentsMargins(10, 0, 0, 0)
        pad_layout.addWidget(line)
        pad_layout.addLayout(extended_layout)

        self.extended_widget = QWidget(self)
        self.extended_widget.setVisible(False)
        self.extended_widget.setLayout(pad_layout)
        layout.addWidget(self.extended_widget)

        self.generate_button = _create_generate_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.generate_button.clicked.connect(control.generate)

        self.add_pose_button = _create_add_pose_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.custom_checkbox = QCheckBox(self.extended_widget)
        self.custom_checkbox.setText(_("Use custom values"))
        self.custom_checkbox.setChecked(control.use_custom_strength)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.custom_checkbox, stretch=1)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.add_pose_button)
        extended_layout.addLayout(actions_layout)

        self.strength_slider = QSlider(self.extended_widget)
        self.strength_slider.setOrientation(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 75)
        self.strength_slider.setValue(control.strength)
        self.strength_slider.setSingleStep(1)
        self.strength_slider.setPageStep(10)
        self.strength_label = QLabel("1.0", self.extended_widget)

        self.range_label = QLabel(_("Range") + ":", self.extended_widget)
        self.range_slider = IntervalSlider(
            low=0, high=20, minimum=0, maximum=20, parent=self.extended_widget
        )
        self.range_slider.intervalChanged.connect(self._set_range)
        self.range_start_label = QLabel("0.00", self.extended_widget)
        self.range_end_label = QLabel("1.00", self.extended_widget)

        # Secondary parameter for NAI modes (vibe: information extracted,
        # precise reference: fidelity), hidden otherwise.
        self.param2_name_label = QLabel("", self.extended_widget)
        self.param2_slider = QSlider(self.extended_widget)
        self.param2_slider.setOrientation(Qt.Orientation.Horizontal)
        self.param2_slider.setRange(0, 100)
        self.param2_slider.setValue(control.param2)
        self.param2_slider.setSingleStep(1)
        self.param2_slider.setPageStep(10)
        self.param2_label = QLabel("1.00", self.extended_widget)

        slider_layout = QGridLayout()
        slider_layout.setSpacing(8)
        slider_layout.addWidget(QLabel(_("Strength") + ":"), 0, 0)
        slider_layout.addWidget(self.strength_slider, 0, 2)
        slider_layout.addWidget(self.strength_label, 0, 3)
        slider_layout.addWidget(self.range_label, 1, 0)
        slider_layout.addWidget(self.range_start_label, 1, 1)
        slider_layout.addWidget(self.range_slider, 1, 2)
        slider_layout.addWidget(self.range_end_label, 1, 3)
        slider_layout.addWidget(self.param2_name_label, 2, 0)
        slider_layout.addWidget(self.param2_slider, 2, 2)
        slider_layout.addWidget(self.param2_label, 2, 3)
        extended_layout.addLayout(slider_layout)

        self._update_visibility()
        self._update_pose_utils()
        self._update_strength()
        self._update_range()
        self._update_param2()
        self._update_custom_values()

        self._connections = [
            bind_combo(control, "mode", self.mode_select),
            bind_combo(control, "layer_id", self.layer_select),
            bind_toggle(control, "use_custom_strength", self.custom_checkbox),
            bind(control, "preset_value", self.preset_slider, "value"),
            bind(control, "strength", self.strength_slider, "value"),
            bind(control, "param2", self.param2_slider, "value"),
            control.use_custom_strength_changed.connect(self._update_custom_values),
            control.strength_changed.connect(self._update_strength),
            control.start_changed.connect(self._update_range),
            control.end_changed.connect(self._update_range),
            control.param2_changed.connect(self._update_param2),
            control.has_active_job_changed.connect(self._update_job_active),
            control.error_text_changed.connect(self._set_error),
            control.is_supported_changed.connect(self._update_visibility),
            control.has_range_changed.connect(self._update_visibility),
            control.can_generate_changed.connect(self._update_visibility),
            control.mode_changed.connect(self._update_visibility),
            control.mode_changed.connect(self._update_custom_values),
            control.is_pose_vector_changed.connect(self._update_pose_utils),
            root.active_model.style_changed.connect(self._update_visibility),
            root.active_model.style_changed.connect(self._update_modes),
        ]

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)

    def _update_modes(self):
        is_nai = root.active_model.arch is Arch.nai
        modes = [m for m in ControlMode if not m.is_internal and m.is_nai == is_nai]
        if self._control.mode not in modes:
            modes.insert(0, self._control.mode)  # keep showing the stored mode
        with SignalBlocker(self.mode_select):
            self.mode_select.clear()
            for mode in modes:
                icon = theme.icon(f"control-{mode.name}")
                self.mode_select.addItem(icon, mode.text, mode)
            index = self.mode_select.findData(self._control.mode)
            if index >= 0:
                self.mode_select.setCurrentIndex(index)

    def _layer_thumbnail(self, layer) -> QImage | None:
        try:
            # Use the UNCLAMPED layer bounds for the aspect ratio — vibe/precise
            # reference layers may extend beyond the canvas and are sent uncropped,
            # so the preview must reflect the full layer, not the canvas crop.
            bounds = getattr(layer, "full_bounds", None) or layer.bounds
            extent = bounds.extent if not bounds.is_zero else root.active_model.document.extent
            scale = min(_thumbnail_extent.width / max(extent.width, 1),
                        _thumbnail_extent.height / max(extent.height, 1))
            size = Extent(max(int(extent.width * scale), 1), max(int(extent.height * scale), 1))
            return layer.thumbnail(size)
        except Exception:
            return None

    def _update_layers(self):
        layers = reversed(root.active_model.layers.images)
        is_nai_mode = self._control.mode.is_nai
        with SignalBlocker(self.layer_select):
            self.layer_select.clear()
            index = -1
            if is_nai_mode:  # NAI modes may use the flattened canvas as source
                self.layer_select.addItem(theme.icon("workspace-generation"), "整张画布", QUuid())
                canvas_thumb = self._layer_thumbnail(root.active_model.layers.root)
                self.layer_select.setItemData(
                    0, _thumbnail_tooltip(canvas_thumb, "整张画布"), Qt.ItemDataRole.ToolTipRole
                )
                if self._control.layer_id.isNull():
                    index = 0
            for layer in layers:
                self.layer_select.addItem(layer.name, layer.id)
                item = self.layer_select.count() - 1
                self.layer_select.setItemData(
                    item,
                    _thumbnail_tooltip(self._layer_thumbnail(layer), layer.name),
                    Qt.ItemDataRole.ToolTipRole,
                )
                if layer.id == self._control.layer_id:
                    index = item
            if index == -1 and self._control_list and self._control in self._control_list:
                self.remove()
            elif index >= 0:
                self.layer_select.setCurrentIndex(index)

    def remove(self):
        assert self._control_list is not None
        self._control_list.remove(self._control)

    def resizeEvent(self, a0: QResizeEvent | None):
        super().resizeEvent(a0)
        self._update_visibility()

    def _add_pose_character(self):
        root.active_model.document.add_pose_character(self._control.layer)

    def _update_visibility(self):
        is_small = self.width() < 420
        is_pose = self._control.mode is ControlMode.pose
        is_edit = root.active_model.arch.supports_edit
        is_nai = self._control.mode.is_nai
        # All NAI modes have per-layer knobs in the extended panel: img2img gets
        # strength + noise, vibe gets strength + information extracted, precise
        # references get strength + fidelity.
        has_param2 = is_nai

        def controls():
            self.layer_select.setVisible(self._control.is_supported)
            self.preset_slider.setVisible(self._control.is_supported and not is_edit and not is_nai)
            self.expand_button.setVisible(self._control.is_supported and not is_edit)
            self.generate_button.setVisible(self._control.can_generate and is_small)
            self.generate_tool_button.setVisible(self._control.can_generate and not is_small)
            self.add_pose_button.setVisible(is_pose and is_small)
            self.add_pose_tool_button.setVisible(is_pose and not is_small)
            self.range_label.setVisible(self._control.has_range and not is_nai)
            self.range_slider.setVisible(self._control.has_range and not is_nai)
            self.range_start_label.setVisible(self._control.has_range and not is_nai)
            self.range_end_label.setVisible(self._control.has_range and not is_nai)
            self.custom_checkbox.setVisible(not is_nai)
            self.param2_name_label.setVisible(has_param2)
            self.param2_slider.setVisible(has_param2)
            self.param2_label.setVisible(has_param2)
            if has_param2:
                if self._control.mode is ControlMode.nai_base:
                    self.param2_name_label.setText("噪声:")
                elif self._control.mode is ControlMode.nai_vibe:
                    self.param2_name_label.setText("信息提取度:")
                else:
                    self.param2_name_label.setText("保真度:")
            if not self._control.is_supported or is_edit:
                self.expand_button.setChecked(False)

        def error():
            self.error_text.setVisible(not self._control.is_supported)

        if self._control.is_supported:
            error()
            controls()
        else:  # always hide things to hide first to make space in the layout
            controls()
            error()

    def _update_range(self):
        self.range_slider.setInterval(int(self._control.start * 20), int(self._control.end * 20))
        self.range_start_label.setText(f"{self._control.start:.2f}")
        self.range_end_label.setText(f"{self._control.end:.2f}")

    def _set_range(self, low: int, high: int):
        self._control.start = low / 20
        self._control.end = high / 20

    def _update_strength(self):
        self.strength_label.setText(
            f"{self._control.strength / ControlLayer.strength_multiplier:.2f}"
        )

    def _update_job_active(self):
        self.generate_button.setEnabled(not self._control.has_active_job)
        self.generate_tool_button.setEnabled(not self._control.has_active_job)
        self.layer_select.setEnabled(not self._control.has_active_job)

    def _update_param2(self):
        self.param2_label.setText(f"{self._control.param2 / 100:.2f}")

    def _update_custom_values(self):
        is_nai = self._control.mode.is_nai  # NAI modes always use direct values
        self.preset_slider.setEnabled(not self._control.use_custom_strength)
        self.strength_slider.setEnabled(self._control.use_custom_strength or is_nai)
        self.range_slider.setEnabled(self._control.use_custom_strength)
        # NAI strength is 0..1 (img2img effectively 0.01..0.99); ControlNet allows 1.5.
        max_strength = ControlLayer.strength_multiplier if is_nai else 75
        if self.strength_slider.maximum() != max_strength:
            self.strength_slider.setMaximum(max_strength)

    def _update_pose_utils(self):
        for button in (self.add_pose_button, self.add_pose_tool_button):
            button.setEnabled(self._control.is_pose_vector)
            button.setToolTip(
                _("Add new character pose to selected layer")
                if self._control.is_pose_vector
                else _("Disabled: selected layer must be a vector layer to add a pose")
            )

    def _toggle_extended(self):
        self.extended_widget.setVisible(self.expand_button.isChecked())

    def _set_error(self, error: str):
        parts = error.split("[", 2)
        self.error_text.setText(parts[0])
        if len(parts) > 1:
            self.error_text.setToolTip(
                _("Required model not found, searching for") + f": {parts[1][:-1]}"
            )


def _create_generate_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText(_("From Image"))
    button.setIcon(theme.icon("control-generate"))
    button.setToolTip(_("Generate control layer from current image"))
    return button


def _create_add_pose_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText(_("Add Skeleton"))
    button.setIcon(theme.icon("add-pose"))
    return button


class ControlListWidget(QWidget):
    _widgets: list[ControlWidget]
    _model: ControlLayerList
    _model_connections: list[QMetaObject.Connection]

    changed = pyqtSignal()

    def __init__(self, model: ControlLayerList, parent=None):
        super().__init__(parent)
        self._model = model
        self._widgets = []
        self._model_connections = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: ControlLayerList):
        if self._model != model:
            Binding.disconnect_all(self._model_connections)
            self._model = model
            while len(self._widgets) > 0:
                self._remove_widget(self._widgets[0])
            for control in self._model:
                self._add_widget(control)
            self._model_connections = [
                model.added.connect(self._add_widget),
                model.removed.connect(self._remove_widget),
            ]

    def _add_widget(self, control: ControlLayer):
        widget = ControlWidget(self._model, control, self)
        self._widgets.append(widget)
        self._layout.addWidget(widget)

    def _remove_widget(self, widget: ControlWidget | ControlLayer):
        if isinstance(widget, ControlLayer):
            widget = next(w for w in self._widgets if w._control == widget)
        self._widgets.remove(widget)
        widget.disconnect_all()
        widget.deleteLater()
