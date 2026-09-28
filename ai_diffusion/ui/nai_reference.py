from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PyQt5.QtCore import (
    QEvent,
    QMimeData,
    QObject,
    QSignalBlocker,
    Qt,
    QTimer,
    QUuid,
    pyqtBoundSignal,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QDrag, QDropEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..image import Extent, Image
from ..model.nai_reference import NaiReferenceLibrary
from ..util import client_logger as log
from ..util import ensure
from . import theme

if TYPE_CHECKING:
    from ..model.model import DocumentModel

_id_role = int(Qt.ItemDataRole.UserRole)
_kind_role = _id_role + 1
_drag_format = "application/x-ai-diffusion-nai-reference"


@dataclass
class ReferenceDrop:
    paths: list[Path]
    image: QImage | None = None

    @staticmethod
    def from_mime(mime: QMimeData | None):
        if mime is None:
            return None
        if mime.hasFormat("application/x-krita-node-internal-pointer"):
            return None
        paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        if paths:
            return ReferenceDrop(paths)
        if mime.hasImage():
            image = mime.imageData()
            if isinstance(image, QPixmap):
                image = image.toImage()
            if isinstance(image, QImage) and not image.isNull():
                return ReferenceDrop([], QImage(image))
        return None


class NaiReferenceTree(QTreeWidget):
    files_dropped = pyqtSignal(object, object)
    item_moved = pyqtSignal(str, object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(18)
        self.setUniformRowHeights(True)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(True)
        self.setExpandsOnDoubleClick(False)
        cast(pyqtBoundSignal, self.itemClicked).connect(self._toggle_folder)

    def _toggle_folder(self, item, _column):
        if item.data(0, _kind_role) == "folder":
            item.setExpanded(not item.isExpanded())

    def scrollTo(self, index, hint=QAbstractItemView.ScrollHint.EnsureVisible):
        parent = index.parent()
        while parent.isValid():
            if not self.isExpanded(parent):
                return
            parent = parent.parent()
        super().scrollTo(index, hint)

    def mimeTypes(self):
        return [*super().mimeTypes(), _drag_format]

    @staticmethod
    def folder_for_item(item: QTreeWidgetItem | None, into=True):
        if item is None:
            return None
        if into and item.data(0, _kind_role) == "folder":
            return item.data(0, _id_role)
        parent = item.parent()
        return parent.data(0, _id_role) if parent is not None else None

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        data = {
            "kind": item.data(0, _kind_role),
            "id": item.data(0, _id_role).toString(),
        }
        mime = QMimeData()
        mime.setData(_drag_format, json.dumps(data).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon(0).pixmap(20, 20))
        drag.exec(Qt.DropAction.MoveAction)

    def _is_internal(self, event):
        return event.source() is self and event.mimeData().hasFormat(_drag_format)

    def dragEnterEvent(self, *events, **kwargs):
        event = events[0] if events else kwargs.get("e")
        if event is None:
            return
        if self._is_internal(event):
            event.acceptProposedAction()
        elif ReferenceDrop.from_mime(event.mimeData()) is not None:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event is None:
            return
        if self._is_internal(event):
            super().dragMoveEvent(event)
        elif ReferenceDrop.from_mime(event.mimeData()) is not None:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event is None:
            return
        target = self.itemAt(event.pos())
        if self._is_internal(event):
            data = json.loads(bytes(ensure(event.mimeData()).data(_drag_format)).decode("utf-8"))
            into = self.dropIndicatorPosition() == QAbstractItemView.DropIndicatorPosition.OnItem
            folder_id = self.folder_for_item(target, into)
            cast(pyqtBoundSignal, self.item_moved).emit(data["kind"], QUuid(data["id"]), folder_id)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        elif payload := ReferenceDrop.from_mime(event.mimeData()):
            cast(pyqtBoundSignal, self.files_dropped).emit(payload, self.folder_for_item(target))
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()


class NaiReferenceDialog(QDialog):
    def __init__(
        self, model: DocumentModel | None = None, parent=None, selected_id: QUuid | None = None
    ):
        super().__init__(parent)
        self._library = (
            model.nai_references if model is not None else NaiReferenceLibrary.instance()
        )
        self._choosing = selected_id is not None
        self._importing = False
        self._tree_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self.selected_id: QUuid | None = None
        self.setWindowTitle("NAI 素材库")
        self.setWindowIcon(theme.icon("nai-album"))
        self.resize(660, 440)
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.items = NaiReferenceTree(self)
        self.items.currentItemChanged.connect(self._preview)
        self.items.itemDoubleClicked.connect(self._activate)
        self.items.files_dropped.connect(self.import_drop)
        self.items.item_moved.connect(self._move_item)
        self.items.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.items.customContextMenuRequested.connect(self._context_menu)
        row.addWidget(self.items, 3)
        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(192, 192)
        self.preview.setWordWrap(True)
        row.addWidget(self.preview, 2)
        layout.addLayout(row, 1)
        self.error = QLabel(self._library.error, self)
        self.error.setWordWrap(True)
        self.error.setVisible(bool(self._library.error))
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("导入", self)
        self.folder_button = QPushButton("新建分类", self)
        self.rename_button = QPushButton("重命名", self)
        self.remove_button = QPushButton("删除", self)
        self.expand_button = QPushButton("全部展开", self)
        for button, callback in (
            (self.import_button, self._import_files),
            (self.folder_button, self._new_folder),
            (self.rename_button, self._rename),
            (self.remove_button, self._remove),
            (self.expand_button, self._toggle_expansion),
        ):
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        ensure(footer.button(QDialogButtonBox.StandardButton.Close)).setText("关闭")
        self._use_button = None
        if self._choosing:
            self._use_button = footer.addButton("使用", QDialogButtonBox.ButtonRole.AcceptRole)
        footer.accepted.connect(self.accept)
        footer.rejected.connect(self.reject)
        buttons.addWidget(footer)
        layout.addLayout(buttons)
        self.items.itemExpanded.connect(self._update_expand_button)
        self.items.itemCollapsed.connect(self._update_expand_button)
        self._connection = self._library.changed.connect(self._refresh)
        self._refresh(selected_id, expanded_folders=self._library.load_expanded_folders())

    def done(self, a0: int):
        result = a0
        if self._connection is not None:
            try:
                self._library.save_expanded_folders(self._expanded_folders())
            except Exception:
                log.exception("Could not save NAI folder expansion state")
            QObject.disconnect(self._connection)
            self._connection = None
        super().done(result)

    def accept(self):
        self.selected_id = self._current_id()
        if self._choosing and self.selected_id is None:
            return
        super().accept()

    def _current_key(self):
        item = self.items.currentItem()
        if item is None:
            return None
        return item.data(0, _kind_role), item.data(0, _id_role).toString()

    def _current_id(self):
        item = self.items.currentItem()
        if item is not None and item.data(0, _kind_role) == "reference":
            return item.data(0, _id_role)
        return None

    def _expanded_folders(self):
        return {
            key[1]
            for key, item in self._tree_items.items()
            if key[0] == "folder" and item.isExpanded()
        }

    def _expandable_folders(self):
        return [
            item
            for key, item in self._tree_items.items()
            if key[0] == "folder" and item.childCount() > 0
        ]

    def _update_expand_button(self, *_):
        folders = self._expandable_folders()
        collapse = bool(folders) and all(item.isExpanded() for item in folders)
        self.expand_button.setEnabled(bool(folders))
        self.expand_button.setText("全部收起" if collapse else "全部展开")

    def _toggle_expansion(self):
        expand = not all(item.isExpanded() for item in self._expandable_folders())
        blocker = QSignalBlocker(self.items)
        for key, item in self._tree_items.items():
            if key[0] == "folder":
                item.setExpanded(expand)
        blocker.unblock()
        self._update_expand_button()

    def _refresh(self, selected_id=None, folder_id=None, *, expanded_folders=None):
        if self._importing:
            return
        selected = self._current_key()
        if selected_id is not None:
            selected = ("reference", selected_id.toString())
        elif folder_id is not None:
            selected = ("folder", folder_id.toString())
        expanded = self._expanded_folders() if expanded_folders is None else set(expanded_folders)
        scrollbar = ensure(self.items.verticalScrollBar())
        scroll = scrollbar.value()
        blocker = QSignalBlocker(self.items)
        self.items.clear()
        self._tree_items.clear()
        for folder in self._library.folders:
            item = QTreeWidgetItem([" ".join(folder.name.splitlines())])
            item.setData(0, _id_role, folder.id)
            item.setData(0, _kind_role, "folder")
            item.setIcon(0, theme.icon("nai-folder"))
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
            )
            item.setForeground(0, QBrush(QColor("#d9b56f" if theme.is_dark else "#976824")))
            item.setToolTip(0, folder.name)
            self._tree_items[("folder", folder.id.toString())] = item
        for folder in self._library.folders:
            item = self._tree_items[("folder", folder.id.toString())]
            parent = (
                self._tree_items.get(("folder", folder.parent_id.toString()))
                if folder.parent_id
                else None
            )
            if parent is not None:
                parent.addChild(item)
            else:
                self.items.addTopLevelItem(item)
        for reference in self._library:
            item = QTreeWidgetItem([" ".join(reference.name.splitlines())])
            item.setData(0, _id_role, reference.id)
            item.setData(0, _kind_role, "reference")
            item.setIcon(0, theme.icon("nai-album"))
            item.setFlags(
                cast(
                    Qt.ItemFlags,
                    (item.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsDropEnabled,
                )
            )
            item.setToolTip(0, reference.name)
            parent = (
                self._tree_items.get(("folder", reference.folder_id.toString()))
                if reference.folder_id
                else None
            )
            if parent is not None:
                parent.addChild(item)
            else:
                self.items.addTopLevelItem(item)
            self._tree_items[("reference", reference.id.toString())] = item
        if selected is not None and (item := self._tree_items.get(selected)) is not None:
            self.items.setCurrentItem(item)
            parent = item.parent()
            while (
                parent is not None
                and expanded_folders is None
                and (selected_id is not None or folder_id is not None)
            ):
                expanded.add(parent.data(0, _id_role).toString())
                parent = parent.parent()
        for key, item in self._tree_items.items():
            if key[0] == "folder":
                item.setExpanded(key[1] in expanded)
        scrollbar.setValue(scroll)
        blocker.unblock()
        self._update_expand_button()
        self._preview()

    def _preview(self, *_):
        self.preview.clear()
        reference_id = self._current_id()
        self.rename_button.setEnabled(self.items.currentItem() is not None)
        self.remove_button.setEnabled(self.items.currentItem() is not None)
        if self._use_button is not None:
            self._use_button.setEnabled(reference_id is not None)
        if reference_id is not None:
            try:
                image = self._library.image(reference_id)
                self.preview.setPixmap(Image.scale_to_fit(image, Extent(240, 340)).to_pixmap())
            except Exception as error:
                self.preview.setText(str(error))

    def _activate(self, item, _column):
        if self._choosing and item.data(0, _kind_role) == "reference":
            self.accept()

    def _new_folder(self):
        self._create_folder(self.items.folder_for_item(self.items.currentItem()))

    def _create_folder(self, parent_id):
        name, accepted = QInputDialog.getText(self, "新建分类", "名称")
        if accepted and name.strip():
            try:
                folder = self._library.add_folder(name, parent_id)
                self._refresh(folder_id=folder.id)
            except Exception as error:
                QMessageBox.warning(self, "无法新建分类", str(error))

    def _import_files(self):
        filenames, _filter = QFileDialog.getOpenFileNames(
            self,
            "导入素材",
            "",
            "图像 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.kra *.ora);;所有文件 (*)",
        )
        if filenames:
            folder_id = self.items.folder_for_item(self.items.currentItem())
            self.import_drop(ReferenceDrop([Path(name) for name in filenames]), folder_id)

    def import_drop(self, payload: ReferenceDrop, folder_id=None):
        errors = []
        selected = None
        self._importing = True
        try:
            for path in payload.paths:
                try:
                    selected = self._library.import_file(path, folder_id).id
                except Exception as error:
                    errors.append(f"{path.name}: {error}")
            if payload.image is not None:
                try:
                    selected = self._library.add(
                        Image(payload.image), "拖入图像", folder_id=folder_id
                    ).id
                except Exception as error:
                    errors.append(str(error))
        finally:
            self._importing = False
        self._refresh(selected, folder_id)
        if errors:
            QMessageBox.warning(self, "部分素材未导入", "\n".join(errors))

    def _rename(self):
        key = self._current_key()
        if key is None:
            return
        kind, identity = key[0], QUuid(key[1])
        entry = (
            self._library.find_folder(identity)
            if kind == "folder"
            else self._library.find(identity)
        )
        if entry is None:
            return
        name, accepted = QInputDialog.getText(self, "重命名", "名称", text=entry.name)
        if accepted and name.strip():
            try:
                if kind == "folder":
                    self._library.rename_folder(identity, name)
                else:
                    self._library.rename(identity, name)
            except Exception as error:
                QMessageBox.warning(self, "无法重命名", str(error))

    def _remove(self):
        key = self._current_key()
        if key is None:
            return
        kind, identity = key[0], QUuid(key[1])
        message = (
            "删除此分类？子分类与素材将移到上一级，图片保留。"
            if kind == "folder"
            else "删除此素材？所有作品中的引用都会受影响，且无法撤销。"
        )
        answer = QMessageBox.question(
            self,
            "删除分类" if kind == "folder" else "删除素材",
            message,
            cast(
                QMessageBox.StandardButtons,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ),
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                if kind == "folder":
                    self._library.remove_folder(identity)
                else:
                    self._library.remove(identity)
            except Exception as error:
                QMessageBox.warning(self, "无法删除", str(error))

    def _move_item(self, kind, identity, folder_id):
        try:
            if kind == "folder":
                self._library.move_folder(identity, folder_id)
                self._refresh(folder_id=identity)
            else:
                self._library.move_reference(identity, folder_id)
                self._refresh(selected_id=identity)
        except Exception as error:
            QMessageBox.warning(self, "无法移动", str(error))

    def _context_menu(self, position):
        item = self.items.itemAt(position)
        self.items.setCurrentItem(item)
        if item is None:
            self.items.clearSelection()
        menu = QMenu(self)
        menu.addAction("导入", self._import_files)
        folder_id = self.items.folder_for_item(item)
        menu.addAction(
            "新建子分类" if folder_id else "新建分类", lambda: self._create_folder(folder_id)
        )
        if folder_id is not None:
            menu.addAction("新建顶层分类", lambda: self._create_folder(None))
        if item is not None:
            menu.addSeparator()
            menu.addAction("重命名", self._rename)
            kind, identity = item.data(0, _kind_role), item.data(0, _id_role)
            if item.parent() is not None:
                menu.addAction("移到顶层", lambda: self._move_item(kind, identity, None))
            menu.addAction("删除", self._remove)
        menu.exec(ensure(self.items.viewport()).mapToGlobal(position))

    def dragEnterEvent(self, a0):
        event = a0
        if event is None:
            return
        if ReferenceDrop.from_mime(event.mimeData()) is not None:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, a0):
        event = a0
        if event is None:
            return
        if payload := ReferenceDrop.from_mime(event.mimeData()):
            self.import_drop(payload, self.items.folder_for_item(self.items.currentItem()))
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()


def show_reference_library(parent=None, payload: ReferenceDrop | None = None):
    dialog = NaiReferenceDialog(parent=parent)
    if payload is not None:
        dialog.import_drop(payload)
    dialog.exec()
    dialog.deleteLater()


class NaiCanvasDropFilter(QObject):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._pending = None

    @staticmethod
    def _is_canvas(widget):
        while isinstance(widget, QWidget):
            if widget.inherits("KisView"):
                return True
            widget = widget.parentWidget()
        return False

    def _clear_pending(self):
        self._pending = None

    def eventFilter(self, a0, a1):
        watched, event = a0, a1
        if event is None:
            return False
        try:
            if (
                event.type() == QEvent.Type.Drop
                and isinstance(event, QDropEvent)
                and isinstance(watched, QWidget)
                and self._is_canvas(watched)
            ):
                self._pending = None
                payload = ReferenceDrop.from_mime(event.mimeData())
                if payload is not None:
                    self._pending = (payload, watched.window())
                    QTimer.singleShot(0, self._clear_pending)
            elif (
                event.type() == QEvent.Type.Show
                and isinstance(watched, QMenu)
                and watched.objectName() == "drop_popup"
                and self._pending is not None
            ):
                payload, parent = self._pending
                self._pending = None
                action = QAction("添加到 NAI 素材库", watched)
                watched.addAction(action)
                action.setIcon(theme.icon("nai-album"))
                action.setObjectName("ai_diffusion_insert_nai_reference")
                watched.insertAction(watched.actions()[0], action)
                action.triggered.connect(
                    lambda: QTimer.singleShot(0, lambda: show_reference_library(parent, payload))
                )
        except Exception:
            self._pending = None
            log.exception("Failed to extend the NAI canvas drop menu")
        return False


def install_canvas_drop_filter():
    application = QApplication.instance()
    if application is None:
        return None
    drop_filter = NaiCanvasDropFilter(application)
    application.installEventFilter(drop_filter)
    return drop_filter
