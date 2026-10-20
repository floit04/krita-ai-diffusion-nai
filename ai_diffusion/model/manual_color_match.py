"""Match the active paint layer to the composite underneath it.

The result goes into a new layer directly above the source: a duplicate of it
(same blend mode, opacity, masks) with its pixels replaced. Cloning is what makes
the operation part of Krita's own undo history - ``Node.addChildNode`` and
``Node.remove`` issue ``KisImageLayerAddCommand`` / ``KisImageLayerRemoveCommand``,
while ``Node.setPixelData`` writes straight into the paint device and records
nothing (see ``libs/libkis/Node.cpp``). So Ctrl+Z removes the matched layer,
Ctrl+Y puts it back without recomputing anything, and the source layer is never
touched.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass

from PyQt5.QtCore import QByteArray, QObject, QTimer, QUuid, pyqtSignal

from .. import eventloop
from ..backend.local_color_match import match_nai_images
from ..image import Bounds, Image, ImageCollection
from ..util import client_logger as log

MAX_PIXELS = 32 * 1024 * 1024

MATCH_LAYER_SUFFIX = " [色彩匹配]"


def uid(node):
    return node.uniqueId().toString()


def path_to(node, target_id, path=()):
    if uid(node) == target_id:
        return path
    for i, child in enumerate(node.childNodes()):
        result = path_to(child, target_id, path + (i,))
        if result is not None:
            return result
    return None


def lower_projection(document, target_id, bounds):
    path = path_to(document.rootNode(), target_id)
    if not path:
        raise ValueError("找不到目标绘画图层。")
    clone = document.clone()
    if clone is None or clone == document:
        raise RuntimeError("无法建立安全的临时文档副本。")
    try:
        clone.setBatchmode(True)
        parent = clone.rootNode()
        for level, index in enumerate(path):
            children = parent.childNodes()
            if index >= len(children):
                raise RuntimeError("复制文档时图层结构发生变化。")
            branch = children[index]
            # Krita childNodes are ordered bottom to top.
            for above in children[index + 1 :]:
                above.setVisible(False)
            if level == len(path) - 1:
                branch.setVisible(False)
            parent = branch
        clone.refreshProjection()
        clone.waitForDone()
        data = clone.pixelData(*bounds)
        if len(data) != bounds.width * bounds.height * 4:
            raise ValueError("下方图层像素读取失败。")
        return Image.from_packed_bytes(data, bounds.extent)
    finally:
        clone.close()


def raw_pixels(node, bounds):
    data = bytes(node.pixelData(*bounds))
    if len(data) != bounds.width * bounds.height * 4:
        raise ValueError("当前图层像素读取失败。")
    return data


def geometry(node):
    b = node.bounds()
    p = node.position()
    return (
        b.x(),
        b.y(),
        b.width(),
        b.height(),
        p.x(),
        p.y(),
        node.colorModel(),
        node.colorDepth(),
        node.colorProfile(),
    )


def topology(document):
    def scan(node):
        return (
            uid(node),
            node.type(),
            node.visible(),
            node.opacity(),
            node.blendingMode(),
            tuple(scan(c) for c in node.childNodes()),
        )

    return scan(document.rootNode())


def eligibility(document, node):
    if node is None or node.type() != "paintlayer":
        return "请在 Krita 图层面板选中一个普通绘画图层。"
    if node.animated():
        return "暂不处理动画图层。"
    if document.colorModel() != "RGBA" or document.colorDepth() != "U8":
        return "当前功能支持 RGB/Alpha、8 位文档，不会自动转换位深。"
    if (node.colorModel(), node.colorDepth(), node.colorProfile()) != (
        document.colorModel(),
        document.colorDepth(),
        document.colorProfile(),
    ):
        return "目标图层与文档色彩空间不一致，暂不处理以免色偏。"
    ancestor = node
    while ancestor is not None:
        if ancestor.locked():
            return "图层或所属组已锁定，请先解锁要处理的绘画图层。"
        if not ancestor.visible():
            return "图层或所属组已隐藏，请先显示要处理的图层。"
        if any(c.type() == "transformmask" and c.visible() for c in ancestor.childNodes()):
            return "图层或所属组含变形蒙版，暂不处理。"
        ancestor = ancestor.parentNode()
    b = node.bounds()
    if b.width() <= 0 or b.height() <= 0:
        return "当前图层为空。"
    return ""


@dataclass
class MatchLayer:
    layer_id: QUuid


class ManualColorMatch(QObject):
    changed = pyqtSignal()

    def __init__(self, model):
        super().__init__(model)
        self._model = weakref.ref(model)
        self._states: dict[str, MatchLayer] = {}
        self._task = None
        self._busy = False
        self._last_status = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _context(self):
        model = self._model()
        if model is None:
            return None
        wrapper = model.document
        if not wrapper.is_valid or not wrapper.is_active:
            return None
        doc = getattr(wrapper, "_doc", None)
        if doc is None:
            return None
        return model, wrapper, doc, doc.activeNode()

    def _match_layer(self, model, node):
        point = self._states.get(uid(node)) if node is not None else None
        if point is None:
            return None
        # updated() drops wrappers for layers Krita removed (e.g. after Ctrl+Z).
        return model.document.layers.updated().find(point.layer_id)

    def status(self):
        try:
            ctx = self._context()
            if ctx is None:
                return False, False, "请先打开并选中要处理的 Krita 文档。"
            model, _wrapper, doc, node = ctx
            layer = self._match_layer(model, node)
            applied = layer is not None
            if self._busy:
                return False, applied, "正在匹配当前图层；切换目标会取消。"
            reason = eligibility(doc, node)
            if reason:
                return False, applied, reason
            if layer is not None:
                return (
                    True,
                    True,
                    (
                        f"已匹配为「{layer.name()}」。Ctrl+Z 撤销，Ctrl+Y 重做，都不需要重新计算；"
                        "再次点击此按钮移除该图层。"
                    ),
                )
            return (
                True,
                False,
                (
                    "在上方新建一层匹配结果，参考下方可见图层，原图层不改动。\n"
                    "新图层是 Krita 普通图层，Ctrl+Z 撤销、Ctrl+Y 重做。"
                ),
            )
        except (RuntimeError, AttributeError):
            return False, False, "当前图层暂不可用。"

    def _poll(self):
        value = self.status()
        if value != self._last_status:
            self._last_status = value
            self.changed.emit()

    def _error(self, text):
        model = self._model()
        if model is not None:
            model.report_error(text)

    def toggle(self, checked=False):
        if self._busy:
            return
        try:
            ctx = self._context()
            if ctx is None:
                return
            model, wrapper, doc, node = ctx
            reason = eligibility(doc, node)
            if reason:
                self._error(reason)
                return
            if layer := self._match_layer(model, node):
                # A native remove command, so this step is undoable as well. The
                # state is kept: undoing the removal brings the same node back and
                # the button returns to "applied" without a new match.
                layer.remove()
                doc.refreshProjection()
                doc.setModified(True)
                return
            self._busy = True
            self.changed.emit()
            self._task = eventloop.run(self._compute(wrapper, doc, node))
        except Exception as error:
            self._busy = False
            log.exception("Current-layer color matching failed")
            self._error(str(error))
        finally:
            self.changed.emit()

    def _current(self, wrapper, doc, key):
        try:
            ctx = self._context()
            if ctx is None or ctx[2] != doc or ctx[3] is None or uid(ctx[3]) != key:
                return None
            node = doc.nodeByUniqueID(ctx[3].uniqueId())
            if node is None or eligibility(doc, node):
                return None
            return node
        except (RuntimeError, AttributeError):
            return None

    def _add_match_layer(self, wrapper, doc, node, bounds, output):
        """Clone the source layer so masks, blend mode and opacity come along, then
        swap in the matched pixels before it enters the layer stack."""
        duplicate = node.duplicate()
        if duplicate is None:
            raise RuntimeError("无法复制目标图层。")
        try:
            if not duplicate.setPixelData(QByteArray(output), *bounds):
                raise RuntimeError("Krita 拒绝写入匹配结果。")
            duplicate.setName(node.name() + MATCH_LAYER_SUFFIX)
            parent = node.parentNode()
            if parent is None:
                raise RuntimeError("找不到目标图层的父级。")
            # This is the undo command; the pixels are already in the node, so
            # redo re-inserts the finished layer instead of recomputing it.
            parent.addChildNode(duplicate, node)
        except Exception:
            duplicate.remove()  # detached, so this is a no-op, but keeps it tidy
            raise
        doc.refreshProjection()
        doc.waitForDone()
        wrapper.layers.update()
        doc.setModified(True)
        return duplicate

    async def _compute(self, wrapper, doc, initial_node):
        key = uid(initial_node)
        try:
            node = self._current(wrapper, doc, key)
            if node is None:
                return
            doc.waitForDone()
            b = node.bounds()
            x, y = max(0, b.x()), max(0, b.y())
            right, bottom = (
                min(doc.width(), b.x() + b.width()),
                min(doc.height(), b.y() + b.height()),
            )
            bounds = Bounds(x, y, max(0, right - x), max(0, bottom - y))
            if bounds.is_zero:
                raise ValueError("当前图层在画布内没有像素。")
            if bounds.width * bounds.height > MAX_PIXELS:
                raise ValueError("目标范围超过 3200 万像素，暂不处理。")
            baseline_geometry = geometry(node)
            baseline_tree = topology(doc)
            size = (doc.width(), doc.height(), doc.currentTime())
            original = raw_pixels(node, bounds)
            if not any(original[3::4]):
                raise ValueError("当前图层在这个范围完全透明。")
            source = Image.from_packed_bytes(QByteArray(original), bounds.extent)
            reference = lower_projection(doc, key, bounds)
            if not any(bytes(reference.data)[3::4]):
                raise ValueError("当前图层下方同位置没有可见像素。")
            images = await match_nai_images(
                ImageCollection([source]),
                reference.to_base64(),
                cancelled=lambda: self._current(wrapper, doc, key) is None,
            )
            node = self._current(wrapper, doc, key)
            if node is None:
                return
            doc.waitForDone()
            if geometry(node) != baseline_geometry or size != (
                doc.width(),
                doc.height(),
                doc.currentTime(),
            ):
                raise RuntimeError("计算期间图层位置、尺寸或画布发生变化，已取消。")
            if topology(doc) != baseline_tree or raw_pixels(node, bounds) != original:
                raise RuntimeError("计算期间图层内容或结构发生变化，已取消。")
            if len(images) != 1 or images[0] is source:
                raise RuntimeError("颜色匹配失败，详情见插件日志。")
            corrected = images[0]
            if corrected.extent != source.extent:
                raise RuntimeError("匹配结果尺寸异常，未改动图层。")
            output = bytes(corrected.data)
            if len(output) != len(original) or output[3::4] != original[3::4]:
                raise RuntimeError("匹配结果透明度异常，未改动图层。")
            patched = bytearray(output)
            for offset, alpha in enumerate(original[3::4]):
                if alpha == 0:
                    start = offset * 4
                    patched[start : start + 4] = original[start : start + 4]
            output = bytes(patched)
            layer = self._add_match_layer(wrapper, doc, node, bounds, output)
            self._states[key] = MatchLayer(layer.uniqueId())
            log.info("Matched paint layer into %s", layer.name())
        except asyncio.CancelledError:
            pass
        except Exception as error:
            log.exception("Current-layer color match failed or cancelled")
            self._error(str(error))
        finally:
            self._busy = False
            self._task = None
            self.changed.emit()
