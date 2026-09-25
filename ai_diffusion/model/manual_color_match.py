"""Match the active paint layer to the composite underneath it.

No generation history or automatic processing. Raw backups are kept on disk;
Node.setPixelData does not create a native Krita undo command.
Adapted from the user-provided 2026-09-25 handoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PyQt5.QtCore import QByteArray, QObject, QTimer, pyqtSignal

from .. import eventloop
from ..backend.local_color_match import match_nai_images
from ..image import Bounds, Image, ImageCollection
from ..util import client_logger as log

MAX_PIXELS = 32 * 1024 * 1024


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


def digest(data):
    return hashlib.sha256(data).hexdigest()


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
class RestorePoint:
    bounds: Bounds
    geometry: tuple
    original_path: Path
    original_hash: str
    corrected_hash: str
    document_size: tuple


class ManualColorMatch(QObject):
    changed = pyqtSignal()

    def __init__(self, model):
        super().__init__(model)
        self._model = weakref.ref(model)
        self._states: dict[str, RestorePoint] = {}
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

    def status(self):
        try:
            ctx = self._context()
            if ctx is None:
                return False, False, "请先打开并选中要处理的 Krita 文档。"
            _model, _wrapper, doc, node = ctx
            point = self._states.get(uid(node)) if node is not None else None
            applied = point is not None and point.geometry == geometry(node)
            if self._busy:
                return False, applied, "正在匹配当前图层；切换目标会取消写回。"
            reason = eligibility(doc, node)
            if reason:
                return False, applied, reason
            if applied:
                return (
                    True,
                    True,
                    "已匹配，点击恢复原色。修改过像素会拒绝覆盖并保留备份；不支持 Ctrl+Z。",
                )
            return (
                True,
                False,
                "匹配当前绘画图层，参考下方同位置；再次点击恢复原色。不支持 Ctrl+Z，请先复制图层。",
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
            _model, wrapper, doc, node = ctx
            reason = eligibility(doc, node)
            if reason:
                self._error(reason)
                return
            key = uid(node)
            if key in self._states:
                self._restore(doc, node, self._states[key])
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

    def _snapshot(self, node, bounds, data, source, doc):
        folder = Path(os.environ["LOCALAPPDATA"]) / "KritaColorMatch" / "layer-snapshots"
        folder /= datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "original.bgra"
        path.write_bytes(data)
        (folder / "original.png").write_bytes(bytes(source.to_bytes()))
        metadata = {
            "node_id": uid(node),
            "node_name": node.name(),
            "bounds": list(bounds),
            "document": doc.fileName(),
            "format": "BGRA U8 straight alpha",
            "profile": node.colorProfile(),
            "sha256": digest(data),
            "note": "PNG is a viewable copy; BGRA is the exact restore source.",
        }
        (folder / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def _write(self, doc, node, bounds, output, rollback):
        try:
            if not node.setPixelData(QByteArray(output), *bounds):
                raise RuntimeError("Krita 拒绝写入图层像素。")
            doc.refreshProjection()
            doc.waitForDone()
            if raw_pixels(node, bounds) != output:
                raise RuntimeError("写回校验失败。")
            doc.setModified(True)
        except Exception:
            if not node.setPixelData(QByteArray(rollback), *bounds):
                log.error("Automatic rollback failed; source snapshot retained")
            doc.refreshProjection()
            doc.setModified(True)
            raise

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
            backup = self._snapshot(node, bounds, original, source, doc)
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
                raise RuntimeError("计算期间图层位置、尺寸或画布发生变化，已取消写回。")
            if topology(doc) != baseline_tree or raw_pixels(node, bounds) != original:
                raise RuntimeError("计算期间图层内容或结构发生变化，已保留新修改，不写回。")
            if len(images) != 1 or images[0] is source:
                raise RuntimeError("颜色匹配失败，当前图层未改动，详情见插件日志。")
            corrected = images[0]
            if corrected.extent != source.extent:
                raise RuntimeError("匹配结果尺寸异常，当前图层未改动。")
            output = bytes(corrected.data)
            if len(output) != len(original) or output[3::4] != original[3::4]:
                raise RuntimeError("匹配结果透明度异常，当前图层未改动。")
            patched = bytearray(output)
            for offset, alpha in enumerate(original[3::4]):
                if alpha == 0:
                    start = offset * 4
                    patched[start : start + 4] = original[start : start + 4]
            output = bytes(patched)
            self._write(doc, node, bounds, output, original)
            self._states[key] = RestorePoint(
                bounds, geometry(node), backup, digest(original), digest(output), size
            )
            log.info("Matched paint layer; source backup %s", backup)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            log.exception("Current-layer color match failed or cancelled; source snapshot retained")
            self._error(str(error))
        finally:
            self._busy = False
            self._task = None
            self.changed.emit()

    def _restore(self, doc, node, point):
        key = uid(node)
        doc.waitForDone()
        if (
            geometry(node) != point.geometry
            or (doc.width(), doc.height(), doc.currentTime()) != point.document_size
        ):
            self._states.pop(key, None)
            raise RuntimeError(
                "图层已移动或变换，不覆盖后续修改。原始备份：" + str(point.original_path.parent)
            )
        current = raw_pixels(node, point.bounds)
        if digest(current) != point.corrected_hash:
            self._states.pop(key, None)
            raise RuntimeError(
                "图层像素已修改，不用旧原图覆盖。原始备份：" + str(point.original_path.parent)
            )
        original = point.original_path.read_bytes()
        if digest(original) != point.original_hash:
            raise RuntimeError("原始备份校验失败，未改动图层。")
        self._write(doc, node, point.bounds, original, current)
        self._states.pop(key, None)
        self.changed.emit()
