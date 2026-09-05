from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, cast
from uuid import uuid4
from weakref import WeakValueDictionary

import krita
from krita import Krita
from PyQt5.QtCore import QByteArray, QObject, QTimer, pyqtSignal

from .image import Bounds, Extent, Image, Mask
from .layer import Layer, LayerManager, LayerType
from .localization import translate as _
from .pose import Pose
from .util import acquire_elements

# Name of the reusable overlay layer for painting the inpaint selection. Also used
# to find the layer again after a plugin reload, so it must not change.
SELECTION_PAINT_LAYER_NAME = "AI 重绘选区 (涂抹)"

# Name of the vector layer holding the focused-inpaint box, found again the same
# way as the mask overlay, so it must not change either.
FOCUS_BOX_LAYER_NAME = "AI 聚焦框"


class SelectionModifiers(NamedTuple):
    feather_rel: float = 0.0
    feather_min_px: int = 0
    pad_rel: float = 0.0
    pad_offset_px: int = 0
    size_min_px: int = 0
    multiple: int = 8
    square: bool = False
    invert: bool = False


class Document(QObject):
    """Document interface. Used as placeholder when there is no open Document in Krita."""

    selection_bounds_changed = pyqtSignal()
    current_time_changed = pyqtSignal()

    _layers: LayerManager

    def __init__(self):
        super().__init__()
        self._layers = LayerManager(None)

    @property
    def extent(self):
        return Extent(0, 0)

    @property
    def filename(self) -> str:
        return ""

    def check_color_mode(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        return True, None

    def create_mask_from_selection(
        self, mod: SelectionModifiers
    ) -> tuple[Mask, Bounds] | tuple[None, None]:
        raise NotImplementedError

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None
    ) -> Image:
        raise NotImplementedError

    def resize(self, extent: Extent):
        raise NotImplementedError

    def resize_canvas(self, width: int, height: int):
        """Resize the underlying canvas if supported by the implementation."""

    def annotate(self, key: str, value: QByteArray):
        pass

    def find_annotation(self, key: str) -> QByteArray | None:
        return None

    def remove_annotation(self, key: str):
        pass

    def add_pose_character(self, layer: Layer):
        raise NotImplementedError

    def import_animation(self, files: list[Path], offset: int = 0):
        raise NotImplementedError

    def start_selection_painting(self):
        """Enter paint-selection mode (see KritaDocument). No-op by default."""

    def stop_selection_painting(self):
        """Leave paint-selection mode (see KritaDocument). No-op by default."""

    def discard_selection_painting(self):
        """Delete the painted mask overlay (see KritaDocument). No-op by default."""

    def start_focus_box(self, bounds: Bounds):
        """Create/show the focus box overlay (see KritaDocument). No-op by default."""

    def hide_focus_box(self):
        """Hide the focus box, keeping it (see KritaDocument). No-op by default."""

    def stop_focus_box(self):
        """Delete the focus box overlay (see KritaDocument). No-op by default."""

    def set_focus_box_bounds(self, bounds: Bounds):
        """Move/resize the focus box (see KritaDocument). No-op by default."""

    @property
    def focus_box_layer(self) -> Layer | None:
        """The vector layer holding the focus box (see KritaDocument)."""
        return None

    @property
    def focus_box_bounds(self) -> Bounds | None:
        """The focus box in document pixels (see KritaDocument)."""
        return None

    @property
    def selection_paint_layer(self) -> Layer | None:
        """The overlay layer holding the painted inpaint mask (see KritaDocument)."""
        return None

    @property
    def has_selection_paint_mask(self) -> bool:
        """True while a painted inpaint mask exists (see KritaDocument)."""
        return False

    @property
    def layers(self) -> LayerManager:
        return self._layers

    @property
    def selection_bounds(self) -> Bounds | None:
        return None

    @property
    def resolution(self) -> float:
        return 0.0

    @property
    def playback_time_range(self) -> tuple[int, int]:
        return 0, 0

    @property
    def current_time(self) -> int:
        return 0

    @property
    def is_valid(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return Krita.instance().activeDocument() is None


class KritaDocument(Document):
    """Wrapper around a Krita Document (opened image). Allows to retrieve and modify pixel data.
    Keeps track of selection and current time changes by polling at a fixed interval.
    """

    _instances: WeakValueDictionary[str, KritaDocument] = WeakValueDictionary()

    def __init__(self, krita_document: krita.Document, id: str | None):
        super().__init__()
        self._doc = krita_document
        self._id = id
        if self._id is None:
            self._id = str(uuid4())
            krita_document.setAnnotation(
                "ai_diffusion/document_id",
                "document unique identifier",
                QByteArray(self._id.encode("utf-8")),
            )
        self._instances[self._id] = self
        self._layers = LayerManager(krita_document)
        self._selection_bounds: Bounds | None = None
        self._current_time: int = 0

        self._was_valid = False
        self._poller = QTimer(self)
        self._poller.setInterval(20)
        self._poller.timeout.connect(self._poll)
        self._poller.start()

    @staticmethod
    def _id_from_annotation(doc: krita.Document) -> str | None:
        id = doc.annotation("ai_diffusion/document_id")
        if id and id.size() > 0:
            return str(id.data(), "utf-8")
        return None

    @classmethod
    def active(cls):
        if doc := Krita.instance().activeDocument():
            if doc.activeNode() is None:
                return None
            all_docs = acquire_elements(Krita.instance().documents())
            if doc not in all_docs or not doc.activeNode():
                return None  # document not fully initialized yet
            id = cls._id_from_annotation(doc)
            for other in all_docs:
                other_id = cls._id_from_annotation(other)
                if other != doc and id and other_id == id:
                    id = None  # doc is a copy of other, give it a new ID (see #2164)
                    break
            if id and id in cls._instances:
                cached = cls._instances[id]
                if cached._doc in all_docs:  # don't reuse if the document was closed
                    return cached
            return KritaDocument(doc, id)
        return None

    @classmethod
    def active_instance(cls) -> KritaDocument | None:
        if doc := Krita.instance().activeDocument():
            id = cls._id_from_annotation(doc)
            if id and id in cls._instances:
                return cls._instances[id]
        return None

    @property
    def id(self):
        return self._id

    @property
    def extent(self):
        return Extent(self._doc.width(), self._doc.height())

    @property
    def filename(self):
        return self._doc.fileName()

    @property
    def layers(self):
        return self._layers

    def check_color_mode(self):
        model = self._doc.colorModel()
        msg_fmt = _("Incompatible document: Color {0} must be {1} (current {0}: {2})")
        if model != "RGBA":
            return False, msg_fmt.format("model", "RGB/Alpha", model)
        depth = self._doc.colorDepth()
        if depth != "U8":
            return False, msg_fmt.format("depth", "8-bit integer", depth)
        return True, None

    @property
    def selection_paint_layer(self) -> Layer | None:
        """The overlay layer holding the painted inpaint mask, if it still exists.

        The layer — not the toggle button and not the document selection — is the
        single source of truth for the mask. Deleting it in the layer docker is
        therefore how the user cancels inpainting.
        """
        layers = self._layers.updated()
        layer_id = getattr(self, "_sel_paint_layer_id", None)
        if layer_id is not None:
            if layer := layers.find(layer_id):
                return layer
            self._sel_paint_layer_id = None  # deleted in the layer docker
        # Fall back to the name so the mask survives a plugin reload or reopening
        # the .kra file, where the remembered id is gone.
        layer = next(
            (
                l
                for l in layers.all
                if l.type is LayerType.paint and l.name == SELECTION_PAINT_LAYER_NAME
            ),
            None,
        )
        if layer is not None:
            self._sel_paint_layer_id = layer.id
        return layer

    @property
    def has_selection_paint_mask(self) -> bool:
        """True when the painted overlay is the *effective* inpaint mask.

        A real selection takes priority, and the overlay is then ignored entirely.
        """
        if self._doc.selection() is not None:
            return False
        return self._paint_layer_alpha() is not None

    def _paint_layer_alpha(self) -> bytes | None:
        """Alpha channel of the overlay layer, or None if nothing is painted."""
        from .util import client_logger as log

        layer = self.selection_paint_layer
        if layer is None or layer.bounds.is_zero:
            return None
        try:
            # pixelData, not projectionPixelData: the overlay counts even while it
            # is hidden, and layer opacity must not dilute the mask.
            data = bytes(layer.node.pixelData(0, 0, self._doc.width(), self._doc.height()))
        except Exception as e:
            log.warning(f"selection paint: failed to read mask layer: {e}")
            return None
        alpha = data[3::4]  # BGRA
        return alpha if any(alpha) else None  # painted, then fully erased -> no mask

    def _selection_from_paint_layer(self):
        """Build a Krita selection from the overlay layer's alpha channel."""
        from krita import Selection

        alpha = self._paint_layer_alpha()
        if alpha is None:
            return None
        selection = Selection()
        selection.setPixelData(QByteArray(alpha), 0, 0, self._doc.width(), self._doc.height())
        return selection

    def start_selection_painting(self):
        """Show the paint-selection overlay, mimicking NAI web's "Draw Mask".

        Strokes are painted fully opaque (in NAI-mask blue) while the LAYER is set
        to 50% opacity — overlapping strokes therefore never accumulate. The brush
        switches to Krita's freehand tool with the "d) Ink-3 Gpen" preset.

        The overlay is only ever created or shown here; it is never deleted, and
        its content is the inpaint mask regardless of whether it is visible.
        """
        from .util import client_logger as log

        doc = self._doc
        if getattr(self, "_sel_paint_active", False):
            return  # already painting
        root = doc.rootNode()
        self._sel_paint_prev_node = doc.activeNode()
        self._sel_paint_prev_preset = None
        self._sel_paint_prev_color = None

        # Clear any leftover selection (e.g. from the previous paint-mask run) —
        # Krita brushes are constrained to the active selection, which would
        # otherwise trap all new strokes inside the previous mask.
        if doc.selection() is not None:
            doc.setSelection(None)

        # Reuse the existing overlay (hidden or not) so a mask painted earlier can
        # be adjusted; create a fresh one only after the user deleted it.
        wrapper = self.selection_paint_layer
        layer = wrapper.node if wrapper is not None else None
        if layer is None:
            layer = doc.createNode(SELECTION_PAINT_LAYER_NAME, "paintlayer")
            root.addChildNode(layer, None)  # None = top-most
        layer.setOpacity(128)  # 50%: translucent overlay, strokes never accumulate
        layer.setVisible(True)
        # A plain, independent paint layer: it must never act as a mask on the
        # layers below, only mark the area to redraw. It is also excluded from the
        # captured image, so it never reaches the model.
        layer.setBlendingMode("normal")
        layer.setInheritAlpha(False)
        # Pin above everything, including the plugin's live preview layer.
        if wrapper := self._layers.updated().find(layer.uniqueId()):
            wrapper.move_to_top()
        doc.setActiveNode(layer)
        doc.refreshProjection()
        self._sel_paint_layer_id = layer.uniqueId()
        self._sel_paint_active = True

        window = Krita.instance().activeWindow()
        view = window.activeView() if window else None
        if view is not None:
            try:  # brush preset: hard-edged ink pen, as requested
                self._sel_paint_prev_preset = view.currentBrushPreset()  # type: ignore[attr-defined]
                presets = Krita.instance().resources("preset")
                if gpen := presets.get("d) Ink-3 Gpen"):
                    view.setCurrentBrushPreset(gpen)  # type: ignore[attr-defined]
            except Exception as e:
                log.warning(f"selection paint: could not switch brush preset: {e}")
            try:  # NAI web mask blue (#8286D9). ManagedColor RGBA/U8 is BGRA order.
                from krita import ManagedColor

                self._sel_paint_prev_color = view.foregroundColor()
                color = ManagedColor("RGBA", "U8", "")  # type: ignore[call-arg]
                color.setComponents([0.851, 0.525, 0.510, 1.0])
                view.setForeGroundColor(color)  # type: ignore[attr-defined]
            except Exception as e:
                log.warning(f"selection paint: could not set foreground color: {e}")
        if brush_tool := Krita.instance().action("KritaShape/KisToolBrush"):
            brush_tool.trigger()

    def stop_selection_painting(self):
        """Hide the overlay and restore the brush — the mask itself is kept.

        The button only shows and hides the overlay. The painted mask stays in
        effect while hidden and is only cancelled by deleting the layer.
        """
        from .util import client_logger as log

        doc = self._doc
        was_active = getattr(self, "_sel_paint_active", False)
        self._sel_paint_active = False
        if layer := self.selection_paint_layer:
            layer.hide()
        if not was_active:
            # Nothing was taken over, so there is nothing to give back. Generating
            # calls this on every run, and it must not touch the user's tool then.
            return

        try:  # restore previous layer / brush / color
            prev = getattr(self, "_sel_paint_prev_node", None)
            if prev is not None:
                doc.setActiveNode(prev)
        except Exception as e:
            log.warning(f"selection paint: could not restore previous layer: {e}")
        self._sel_paint_prev_node = None
        window = Krita.instance().activeWindow()
        view = window.activeView() if window else None
        if view is not None:
            try:
                if preset := getattr(self, "_sel_paint_prev_preset", None):
                    view.setCurrentBrushPreset(preset)  # type: ignore[attr-defined]
                if color := getattr(self, "_sel_paint_prev_color", None):
                    view.setForeGroundColor(color)  # type: ignore[attr-defined]
            except Exception as e:
                log.warning(f"selection paint: could not restore brush settings: {e}")
        self._sel_paint_prev_preset = None
        self._sel_paint_prev_color = None
        doc.refreshProjection()

    def discard_selection_painting(self):
        """Delete the mask overlay entirely.

        Hiding it would keep the mask in effect, so the button's off state has to
        remove the layer: that is what makes the next click start from a clean
        one instead of an old mask.
        """
        self.stop_selection_painting()
        if layer := self.selection_paint_layer:
            layer.remove()
        self._sel_paint_layer_id = None
        self._doc.refreshProjection()

    # -- Focused inpaint box --

    def _focus_box_svg(self, bounds: Bounds) -> str:
        """A single rectangle, authored in document pixels like the pose layer:
        Krita maps SVG user units to points 1:1, and `resolution` converts back.

        Filled, with no stroke on purpose. A stroke inflates Shape.boundingBox()
        by half its width, so reading the box back and writing it out again would
        make it creep outwards a little on every poll.
        """
        width, height = self.extent
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
            f'<rect id="nai-focus-box" x="{bounds.x}" y="{bounds.y}"'
            f' width="{bounds.width}" height="{bounds.height}"'
            ' fill="#8286D9" fill-opacity="0.15" stroke="none"/>'
            "</svg>"
        )

    @property
    def focus_box_layer(self) -> Layer | None:
        """The vector layer holding the focus box, if it still exists.

        Like the mask overlay, the layer is the source of truth: deleting it in
        the layer docker is how the user turns focused inpainting off.
        """
        layers = self._layers.updated()
        layer_id = getattr(self, "_focus_box_layer_id", None)
        if layer_id is not None:
            if layer := layers.find(layer_id):
                return layer
            self._focus_box_layer_id = None  # deleted in the layer docker
        layer = next(
            (
                l
                for l in layers.all
                if l.type is LayerType.vector and l.name == FOCUS_BOX_LAYER_NAME
            ),
            None,
        )
        if layer is not None:
            self._focus_box_layer_id = layer.id
        return layer

    @property
    def focus_box_bounds(self) -> Bounds | None:
        """The box the user has dragged on canvas, in document pixels."""
        from .util import client_logger as log

        layer = self.focus_box_layer
        if layer is None:
            return None
        try:
            shapes = acquire_elements(cast(krita.VectorLayer, layer.node).shapes())
        except Exception as e:
            log.warning(f"focus box: could not read the box layer: {e}")
            return None
        if len(shapes) == 0:
            return None
        rect = shapes[0].boundingBox()
        for shape in shapes[1:]:  # a copy-pasted box counts too
            rect = rect.united(shape.boundingBox())
        res = self.resolution
        bounds = Bounds(
            round(rect.x() * res),
            round(rect.y() * res),
            round(rect.width() * res),
            round(rect.height() * res),
        )
        bounds = Bounds.clamp(bounds, self.extent)
        return None if bounds.is_zero else bounds

    def set_focus_box_bounds(self, bounds: Bounds):
        """Replace the box with one of exactly these bounds."""
        layer = self.focus_box_layer
        if layer is None:
            return
        node = cast(krita.VectorLayer, layer.node)
        for shape in acquire_elements(node.shapes()):
            shape.remove()
        node.addShapesFromSvg(self._focus_box_svg(bounds))
        layer.refresh()

    def start_focus_box(self, bounds: Bounds):
        """Show the focus box and hand it to Krita's shape tool to be dragged."""
        from .util import client_logger as log

        layer = self.focus_box_layer
        if layer is None:
            layer = self._layers.create_vector(FOCUS_BOX_LAYER_NAME, self._focus_box_svg(bounds))
            self._focus_box_layer_id = layer.id
        else:
            layer.show()
        layer.move_to_top()
        self._doc.setActiveNode(layer.node)
        self._doc.refreshProjection()
        # The shape tool is what actually moves and scales the box, and it only
        # acts on the active layer — which is why the box layer is activated here.
        if tool := Krita.instance().action("InteractionTool"):
            tool.trigger()
        else:
            log.warning("focus box: shape selection tool not found")

    def hide_focus_box(self):
        """Hide the box without deleting it — it stays in effect while hidden."""
        if layer := self.focus_box_layer:
            layer.hide()

    def stop_focus_box(self):
        """Delete the box layer, which is what turns focused inpainting off."""
        if layer := self.focus_box_layer:
            layer.remove()
        self._focus_box_layer_id = None
        self._doc.refreshProjection()

    def create_mask_from_selection(self, mod: SelectionModifiers):
        # A real selection wins; the painted overlay is then ignored entirely.
        # Otherwise the overlay is the mask — whether it is shown or hidden — and
        # stops applying the moment the user deletes it.
        user_selection = self._doc.selection()
        if not user_selection:
            user_selection = self._selection_from_paint_layer()
        if not user_selection:
            # Fallback: an ACTIVE selection mask counts as the selection, so masks
            # converted from paint layers work without a marching-ants selection.
            active = self._doc.activeNode()
            if active is not None and active.type() == "selectionmask":
                user_selection = active.selection()  # type: ignore[attr-defined]
        if not user_selection:
            return None, None

        if _selection_is_entire_document(user_selection, self.extent):
            return None, None

        selection = user_selection.duplicate()
        original_bounds = Bounds(
            selection.x(), selection.y(), selection.width(), selection.height()
        )
        original_bounds = Bounds.clamp(original_bounds, self.extent)
        if original_bounds.is_zero:
            return None, None
        size_factor = original_bounds.extent.diagonal
        pad_px = max(int(mod.feather_rel * size_factor), mod.feather_min_px)
        pad_px += mod.pad_offset_px
        pad_px += int(mod.pad_rel * size_factor)

        if mod.invert:
            selection.invert()

        bounds = _selection_bounds(selection)
        bounds = Bounds.pad(
            bounds, pad_px, multiple=mod.multiple, min_size=mod.size_min_px, square=mod.square
        )
        bounds = Bounds.clamp(bounds, self.extent)
        if bounds.is_zero:
            return None, None
        data = selection.pixelData(*bounds)
        return Mask(bounds, data), original_bounds

    def get_image(self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None):
        excluded: list[Layer] = []
        if exclude_layers:
            for layer in filter(lambda l: l.is_visible, exclude_layers):
                layer.hide()
                excluded.append(layer)
        # Always refresh: overlays may have been hidden right before this call
        # (generating hides them without a refresh), and the projection updates
        # asynchronously — reading it stale would bake them into the image.
        self._doc.refreshProjection()

        bounds = bounds or Bounds(0, 0, self._doc.width(), self._doc.height())
        # Use Krita's visible projection instead of raw pixel data to avoid
        # color-channel assumptions across document color spaces (the NAI
        # launcher's own Krita bridge does the same). Raw pixelData is in the
        # document's color profile; a generated result pasted back would only
        # match inside the repainted area, showing as a colour shift there.
        img = None
        projection = getattr(self._doc, "projection", None)
        if projection is not None:
            try:
                qimage = projection(*bounds)
                if qimage is not None and not qimage.isNull():
                    img = Image(qimage)
            except Exception as e:
                from .util import client_logger as log

                log.warning(f"projection() capture failed, falling back to pixelData: {e}")
        if img is None:
            img = Image.from_packed_bytes(self._doc.pixelData(*bounds), bounds.extent)

        for layer in excluded:
            layer.show()
        if len(excluded) > 0:
            self._doc.refreshProjection()
        return img

    def resize(self, extent: Extent):
        res = self._doc.resolution()
        self._doc.scaleImage(extent.width, extent.height, res, res, "Bilinear")

    def resize_canvas(self, width: int, height: int):
        self._doc.resizeImage(0, 0, width, height)

    def annotate(self, key: str, value: QByteArray):
        self._doc.setAnnotation(f"ai_diffusion/{key}", f"AI Diffusion Plugin: {key}", value)

    def find_annotation(self, key: str) -> QByteArray | None:
        result = self._doc.annotation(f"ai_diffusion/{key}")
        return result if result.size() > 0 else None

    def remove_annotation(self, key: str):
        self._doc.removeAnnotation(f"ai_diffusion/{key}")

    def add_pose_character(self, layer: Layer):
        assert layer.type is LayerType.vector
        _pose_layers.add_character(cast(krita.VectorLayer, layer.node))

    def import_animation(self, files: list[Path], offset: int = 0):
        success = self._doc.importAnimation([str(f) for f in files], offset, 1)
        if not success and len(files) > 0:
            folder = files[0].parent
            raise RuntimeError(f"Failed to import animation from {folder}")

    @property
    def selection_bounds(self):
        return self._selection_bounds

    @property
    def resolution(self):
        return self._doc.resolution() / 72.0  # KisImage::xRes which is applied to vectors

    @property
    def playback_time_range(self):
        return self._doc.playBackStartTime(), self._doc.playBackEndTime()

    @property
    def current_time(self):
        return self._doc.currentTime()

    @property
    def is_valid(self):
        # can be a document that has been closed, or one that hasn't finished initializing
        return self._doc.activeNode() is not None and self._doc in acquire_elements(
            Krita.instance().documents()
        )

    @property
    def is_active(self):
        return self._doc == Krita.instance().activeDocument()

    def _poll(self):
        if self.is_valid:
            self._was_valid = True
            selection = self._doc.selection()
            selection_bounds = _selection_bounds(selection) if selection else None
            if selection_bounds is not None and selection_bounds.is_zero:
                selection_bounds = None
            if selection_bounds != self._selection_bounds:
                self._selection_bounds = selection_bounds
                self.selection_bounds_changed.emit()

            current_time = self.current_time
            if current_time != self._current_time:
                self._current_time = current_time
                self.current_time_changed.emit()
        elif self._was_valid:
            self._poller.stop()

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, KritaDocument):
            return self._id == other._id
        return False


def _selection_bounds(selection: krita.Selection):
    return Bounds(selection.x(), selection.y(), selection.width(), selection.height())


def _selection_is_entire_document(selection: krita.Selection, extent: Extent):
    bounds = _selection_bounds(selection)
    if bounds.x > 0 or bounds.y > 0:
        return False
    if bounds.width + bounds.x < extent.width or bounds.height + bounds.y < extent.height:
        return False
    mask = selection.pixelData(*bounds)
    return all(x == b"\xff" for x in mask)


class PoseLayers:
    def __init__(self):
        self._layers: dict[str, Pose] = {}
        self._timer = QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def update(self):
        doc = KritaDocument.active_instance()
        if not doc or not doc.is_valid:
            return
        try:
            layer = doc.layers.active
        except Exception:
            return
        if not layer or layer.type is not LayerType.vector:
            return

        layer = cast(krita.VectorLayer, layer.node)
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        self._update(layer, acquire_elements(layer.shapes()), pose, doc.resolution)

    def add_character(self, layer: krita.VectorLayer):
        doc = KritaDocument.active_instance()
        assert doc is not None
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        svg = Pose.create_default(doc.extent, pose.people_count).to_svg()
        shapes = acquire_elements(layer.addShapesFromSvg(svg))
        self._update(layer, shapes, pose, doc.resolution)

    def _update(
        self, layer: krita.VectorLayer, shapes: list[krita.Shape], pose: Pose, resolution: float
    ):
        changes = pose.update(shapes, resolution)  # type: ignore
        if changes:
            shapes = layer.addShapesFromSvg(changes)
            for shape in shapes:
                shape.setZIndex(-1)


_pose_layers = PoseLayers()
