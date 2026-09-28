from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from zipfile import ZipFile

from PyQt5.QtCore import QLockFile, QObject, QUuid, pyqtSignal
from PyQt5.QtGui import QImageReader

from ..image import Extent, Image
from ..util import client_logger as log
from ..util import user_data_dir

if TYPE_CHECKING:
    from ..document import Document


@dataclass(frozen=True)
class NaiReference:
    id: QUuid
    name: str
    extent: Extent
    digest: str
    deleted: bool = False
    folder_id: QUuid | None = None


@dataclass(frozen=True)
class NaiReferenceFolder:
    id: QUuid
    name: str
    parent_id: QUuid | None = None


class NaiReferenceLibrary(QObject):
    changed = pyqtSignal()
    removing = pyqtSignal(QUuid)
    index_key = "nai_references.json"
    _instance: NaiReferenceLibrary | None = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls(user_data_dir / "nai_references")
        return cls._instance

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self._index_path = directory / self.index_key
        self._items: list[NaiReference] = []
        self._folders: list[NaiReferenceFolder] = []
        self._load_error = ""
        self._disk_state = b""
        try:
            if self._index_path.exists():
                self._disk_state = self._index_path.read_bytes()
                items = self._decode_index(self._disk_state)
                folders = self._decode_folders(self._disk_state)
                self._validate_tree(items, folders)
                self._items, self._folders = items, folders
        except Exception as error:
            self._load_error = f"无法读取 NAI 素材库，原文件已保留：{error}"
            log.exception(self._load_error)

    @property
    def error(self):
        return self._load_error

    def __iter__(self):
        return (item for item in self._items if not item.deleted)

    def find(self, reference_id: QUuid) -> NaiReference | None:
        return next((item for item in self._items if item.id == reference_id), None)

    @property
    def folders(self):
        return tuple(self._folders)

    def find_folder(self, folder_id: QUuid | None) -> NaiReferenceFolder | None:
        return next((folder for folder in self._folders if folder.id == folder_id), None)

    def load_expanded_folders(self) -> set[str] | None:
        path = self.directory / "ui_state.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            expanded = state["expanded_folders"]
            if not isinstance(expanded, list) or not all(isinstance(key, str) for key in expanded):
                raise ValueError("Invalid NAI folder expansion state")
            return set(expanded) & {folder.id.toString() for folder in self._folders}
        except FileNotFoundError:
            return None
        except Exception:
            log.exception("Could not load NAI folder expansion state")
            return None

    def save_expanded_folders(self, expanded: set[str]):
        self.directory.mkdir(parents=True, exist_ok=True)
        folder_ids = {folder.id.toString() for folder in self._folders}
        state = {"expanded_folders": sorted(expanded & folder_ids)}
        self._write_atomic(
            self.directory / "ui_state.json", json.dumps(state, indent=2).encode("utf-8")
        )

    @staticmethod
    def _optional_id(value):
        if value is None:
            return None
        identity = QUuid(value)
        if identity.isNull():
            raise ValueError("无效的分类 ID")
        return identity

    @staticmethod
    def _image_key(reference_id: QUuid):
        return f"nai_reference_{reference_id.toString(QUuid.StringFormat.WithoutBraces)}.png"

    @staticmethod
    def _decode_index(data: bytes):
        state = json.loads(data.decode("utf-8"))
        if state.get("version") not in (1, 2):
            raise ValueError("不支持的 NAI 素材库版本")
        items = []
        ids = set()
        for entry in state["items"]:
            reference_id = QUuid(entry["id"])
            width, height = int(entry["width"]), int(entry["height"])
            if reference_id.isNull() or reference_id.toString() in ids or min(width, height) < 1:
                raise ValueError("无效的 NAI 素材记录")
            if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("无效的 NAI 素材校验值")
            ids.add(reference_id.toString())
            items.append(
                NaiReference(
                    reference_id,
                    str(entry["name"]),
                    Extent(width, height),
                    entry["sha256"],
                    bool(entry.get("deleted", False)),
                    NaiReferenceLibrary._optional_id(entry.get("folder_id")),
                )
            )
        return items

    @classmethod
    def _decode_folders(cls, data: bytes):
        state = json.loads(data.decode("utf-8"))
        folders = []
        for entry in state.get("folders", []):
            folder_id = cls._optional_id(entry["id"])
            if folder_id is None or not isinstance(entry["name"], str) or not entry["name"].strip():
                raise ValueError("无效的分类记录")
            folders.append(
                NaiReferenceFolder(
                    folder_id, entry["name"], cls._optional_id(entry.get("parent_id"))
                )
            )
        return folders

    @staticmethod
    def _validate_tree(items: list[NaiReference], folders: list[NaiReferenceFolder]):
        by_id = {folder.id.toString(): folder for folder in folders}
        if len(by_id) != len(folders):
            raise ValueError("分类 ID 重复")
        for folder in folders:
            visited = set()
            current = folder
            while current is not None:
                key = current.id.toString()
                if key in visited:
                    raise ValueError("不能把分类移入自身或其子分类")
                visited.add(key)
                if current.parent_id is None:
                    break
                if current.parent_id.toString() not in by_id:
                    raise ValueError("父分类不存在")
                current = by_id[current.parent_id.toString()]
        for item in items:
            if item.folder_id is not None and item.folder_id.toString() not in by_id:
                raise ValueError("素材所属分类不存在")

    def _check_folder(self, folder_id: QUuid | None):
        if folder_id is not None and self.find_folder(folder_id) is None:
            raise ValueError("分类不存在")

    def _check_folder_name(self, name: str, parent_id: QUuid | None, exclude: QUuid | None = None):
        if not name.strip():
            raise ValueError("分类名称不能为空")
        if any(
            folder.parent_id == parent_id
            and folder.id != exclude
            and folder.name.casefold() == name.strip().casefold()
            for folder in self._folders
        ):
            raise ValueError("同级已有此分类")

    @contextmanager
    def _editing(self):
        if self._load_error:
            raise RuntimeError(self._load_error)
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(self.directory / "library.lock"))
        if not lock.tryLock(0):
            raise RuntimeError("NAI 素材库正在被其他进程使用，请稍后重试")
        try:
            current = self._index_path.read_bytes() if self._index_path.exists() else b""
            if current != self._disk_state:
                raise RuntimeError("NAI 素材库已被其他进程修改，请重启 Krita 后再编辑")
            yield
        finally:
            lock.unlock()

    @staticmethod
    def _write_atomic(path: Path, data: bytes):
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".nai-", delete=False) as output:
                temporary = Path(output.name)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _save(self, items: list[NaiReference], folders: list[NaiReferenceFolder] | None = None):
        folders = self._folders if folders is None else folders
        self._validate_tree(items, folders)
        state = {
            "version": 2,
            "folders": [
                {
                    "id": folder.id.toString(),
                    "name": folder.name,
                    "parent_id": folder.parent_id.toString() if folder.parent_id else None,
                }
                for folder in folders
            ],
            "items": [
                {
                    "id": item.id.toString(),
                    "name": item.name,
                    "width": item.extent.width,
                    "height": item.extent.height,
                    "sha256": item.digest,
                    "deleted": item.deleted,
                    "folder_id": item.folder_id.toString() if item.folder_id else None,
                }
                for item in items
            ],
        }
        data = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
        if self._disk_state and json.loads(self._disk_state).get("version") == 1:
            backup = self.directory / "nai_references.v1-backup.json"
            if not backup.exists():
                self._write_atomic(backup, self._disk_state)
        self._write_atomic(self._index_path, data)
        self._disk_state = data

    def import_document(self, document: Document):
        data = document.find_annotation(self.index_key)
        if data is None:
            return
        try:
            references = self._decode_index(bytes(data))
        except Exception:
            log.exception("Could not read legacy NAI references; document attachments preserved")
            return
        for reference in references:
            if self.find(reference.id) is not None:
                continue
            try:
                image_data = document.find_annotation(self._image_key(reference.id))
                if image_data is None:
                    raise ValueError(f"Missing image for {reference.name}")
                self.add(Image.from_bytes(image_data), reference.name, reference.id)
            except Exception:
                log.exception("Could not migrate NAI reference; document attachment preserved")

    def import_file(self, path: Path, folder_id: QUuid | None = None) -> NaiReference:
        if path.suffix.lower() in (".kra", ".ora"):
            with ZipFile(path) as archive:
                info = archive.getinfo("mergedimage.png")
                if info.file_size > 512 * 1024 * 1024:
                    raise ValueError("合成图像超过 512 MiB，请先导出为 PNG")
                image = Image.from_bytes(archive.read(info))
        else:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            pixels = reader.read()
            if pixels.isNull():
                raise ValueError(f"无法读取图像 {path.name}：{reader.errorString()}")
            image = Image(pixels)
        return self.add(image, path.stem, folder_id=folder_id)

    def add(
        self,
        image: Image,
        name: str,
        reference_id: QUuid | None = None,
        folder_id: QUuid | None = None,
    ) -> NaiReference:
        if min(image.extent) < 1:
            raise ValueError("无法导入空图像")
        data = bytes(image.to_bytes())
        digest = hashlib.sha256(data).hexdigest()
        with self._editing():
            self._check_folder(folder_id)
            existing = (
                self.find(reference_id)
                if reference_id is not None
                else next(
                    (item for item in self._items if item.digest == digest and not item.deleted),
                    None,
                )
            )
            if existing is None and reference_id is None:
                existing = next((item for item in self._items if item.digest == digest), None)
            if existing is not None and existing.digest != digest:
                raise ValueError("NAI 素材 ID 冲突，原素材已保留")
            path = self.directory / f"{digest}.png"
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                self._write_atomic(path, data)
            if existing is not None and not existing.deleted:
                return existing
            item = NaiReference(
                existing.id if existing else reference_id or QUuid.createUuid(),
                name.strip() or "NAI 素材",
                image.extent,
                digest,
                folder_id=folder_id,
            )
            items = list(self._items)
            if existing is not None:
                items[items.index(existing)] = item
            else:
                items.append(item)
            self._save(items)
            self._items = items
        self.changed.emit()
        return item

    def image(self, reference_id: QUuid) -> Image:
        item = self.find(reference_id)
        if item is None or item.deleted:
            raise ValueError("该素材已从本机素材库移除，请重新导入或选择其他素材")
        path = self.directory / f"{item.digest}.png"
        if not path.is_file():
            raise ValueError(f"NAI 素材「{item.name}」的文件缺失，请重新导入原图或迁移素材库")
        return Image.load(path)

    def rename(self, reference_id: QUuid, name: str):
        item = self.find(reference_id)
        if item is None or item.deleted or not name.strip() or item.name == name.strip():
            return
        with self._editing():
            items = list(self._items)
            items[items.index(item)] = replace(item, name=name.strip())
            self._save(items)
            self._items = items
        self.changed.emit()

    def remove(self, reference_id: QUuid):
        item = self.find(reference_id)
        if item is None or item.deleted:
            return
        with self._editing():
            items = list(self._items)
            items[items.index(item)] = replace(item, deleted=True)
            self._save(items)
            self.removing.emit(reference_id)
            self._items = items
            if not any(other.digest == item.digest and not other.deleted for other in items):
                try:
                    (self.directory / f"{item.digest}.png").unlink(missing_ok=True)
                except OSError:
                    log.exception("Could not remove unused NAI reference image")
        self.changed.emit()

    def add_folder(self, name: str, parent_id: QUuid | None = None):
        with self._editing():
            self._check_folder(parent_id)
            self._check_folder_name(name, parent_id)
            folder = NaiReferenceFolder(QUuid.createUuid(), name.strip(), parent_id)
            folders = [*self._folders, folder]
            self._save(self._items, folders)
            self._folders = folders
        self.changed.emit()
        return folder

    def rename_folder(self, folder_id: QUuid, name: str):
        folder = self.find_folder(folder_id)
        if folder is None:
            raise ValueError("分类不存在")
        with self._editing():
            self._check_folder_name(name, folder.parent_id, folder_id)
            folders = [
                replace(item, name=name.strip()) if item.id == folder_id else item
                for item in self._folders
            ]
            self._save(self._items, folders)
            self._folders = folders
        self.changed.emit()

    def move_reference(self, reference_id: QUuid, folder_id: QUuid | None):
        reference = self.find(reference_id)
        if reference is None or reference.deleted:
            raise ValueError("素材不存在")
        if reference.folder_id == folder_id:
            return
        with self._editing():
            self._check_folder(folder_id)
            items = [
                replace(item, folder_id=folder_id) if item.id == reference_id else item
                for item in self._items
            ]
            self._save(items)
            self._items = items
        self.changed.emit()

    def move_folder(self, folder_id: QUuid, parent_id: QUuid | None):
        folder = self.find_folder(folder_id)
        if folder is None:
            raise ValueError("分类不存在")
        if folder.parent_id == parent_id:
            return
        with self._editing():
            self._check_folder(parent_id)
            self._check_folder_name(folder.name, parent_id, folder_id)
            folders = [
                replace(item, parent_id=parent_id) if item.id == folder_id else item
                for item in self._folders
            ]
            self._save(self._items, folders)
            self._folders = folders
        self.changed.emit()

    def remove_folder(self, folder_id: QUuid):
        folder = self.find_folder(folder_id)
        if folder is None:
            return
        with self._editing():
            folders = [
                replace(item, parent_id=folder.parent_id) if item.parent_id == folder_id else item
                for item in self._folders
                if item.id != folder_id
            ]
            items = [
                replace(item, folder_id=folder.parent_id) if item.folder_id == folder_id else item
                for item in self._items
            ]
            self._save(items, folders)
            self._items, self._folders = items, folders
        self.changed.emit()
