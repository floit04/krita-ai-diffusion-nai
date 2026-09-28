import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PyQt5.QtCore import QUuid

from ai_diffusion.image import Extent, Image
from ai_diffusion.model.nai_reference import NaiReferenceLibrary


def test_reference_roundtrip_and_duplicate_import(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    image = Image.create(Extent(16, 24), 0xFFFF0000)
    reference = library.add(image, "First")
    duplicate = library.add(image, "Duplicate")
    assert duplicate.id == reference.id
    assert len(list(library)) == 1

    restored = NaiReferenceLibrary(tmp_path)
    assert not restored.error
    assert restored.find(reference.id) == reference
    assert bytes(restored.image(reference.id).to_bytes()) == bytes(image.to_bytes())


def test_reference_delete_and_reimport_preserves_identity(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    image = Image.create(Extent(8, 8), 0xFF00FF00)
    reference = library.add(image, "Original")
    library.remove(reference.id)
    assert list(library) == []
    assert not (tmp_path / f"{reference.digest}.png").exists()
    with pytest.raises(ValueError):
        library.image(reference.id)

    restored = library.add(image, "Restored")
    assert restored.id == reference.id
    assert not restored.deleted
    assert library.image(restored.id).extent == image.extent


def test_folder_removal_reparents_without_deleting_images(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    parent = library.add_folder("Parent")
    folder = library.add_folder("Folder", parent.id)
    child = library.add_folder("Child", folder.id)
    reference = library.add(Image.create(Extent(8, 8), 0xFF0000FF), "Image", folder_id=folder.id)
    library.remove_folder(folder.id)

    restored = NaiReferenceLibrary(tmp_path)
    restored_child = restored.find_folder(child.id)
    restored_reference = restored.find(reference.id)
    assert restored_child is not None and restored_child.parent_id == parent.id
    assert restored_reference is not None and restored_reference.folder_id == parent.id
    assert restored.image(reference.id).extent == reference.extent


def test_invalid_folder_edits_preserve_the_index(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    parent = library.add_folder("Parent")
    child = library.add_folder("Child", parent.id)
    index = tmp_path / library.index_key
    original = index.read_bytes()
    with pytest.raises(ValueError):
        library.move_folder(parent.id, child.id)
    with pytest.raises(ValueError):
        library.add_folder("parent")
    with pytest.raises(ValueError):
        library.add_folder("Missing parent", QUuid.createUuid())
    assert index.read_bytes() == original
    assert library.find_folder(parent.id) == parent


def test_stale_library_cannot_overwrite_newer_edits(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    stale = NaiReferenceLibrary(tmp_path)
    folder = library.add_folder("Saved")
    original = (tmp_path / library.index_key).read_bytes()
    with pytest.raises(RuntimeError):
        stale.add_folder("Stale")
    assert (tmp_path / library.index_key).read_bytes() == original
    assert NaiReferenceLibrary(tmp_path).find_folder(folder.id) == folder


def test_version_one_migration_keeps_backup_and_image_identity(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    reference = library.add(Image.create(Extent(8, 8), 0xFFFFFFFF), "Legacy")
    index = tmp_path / library.index_key
    state = json.loads(index.read_bytes())
    state["version"] = 1
    del state["folders"]
    for entry in state["items"]:
        del entry["folder_id"]
    original = json.dumps(state).encode("utf-8")
    index.write_bytes(original)

    restored = NaiReferenceLibrary(tmp_path)
    assert not restored.error
    restored.add_folder("New")
    assert (tmp_path / "nai_references.v1-backup.json").read_bytes() == original
    assert json.loads(index.read_bytes())["version"] == 2
    assert restored.find(reference.id) == reference
    assert restored.image(reference.id).extent == reference.extent


def test_corrupt_index_is_preserved_and_blocks_edits(tmp_path: Path):
    index = tmp_path / NaiReferenceLibrary.index_key
    original = b"not valid JSON"
    index.write_bytes(original)
    library = NaiReferenceLibrary(tmp_path)
    assert library.error
    with pytest.raises(RuntimeError):
        library.add_folder("Do not overwrite")
    assert index.read_bytes() == original


def test_folder_expansion_state_ignores_unknown_ids(tmp_path: Path):
    library = NaiReferenceLibrary(tmp_path)
    folder = library.add_folder("Expanded")
    library.save_expanded_folders({folder.id.toString(), QUuid.createUuid().toString()})
    assert NaiReferenceLibrary(tmp_path).load_expanded_folders() == {folder.id.toString()}


def test_reference_dialog_offscreen(tmp_path: Path):
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QImage
from PyQt5.QtWidgets import QApplication, QMenu

app = QApplication([])
from ai_diffusion.image import Extent, Image
from ai_diffusion.model.nai_reference import NaiReferenceLibrary
from ai_diffusion.ui.nai_reference import NaiCanvasDropFilter, NaiReferenceDialog, ReferenceDrop

directory = Path(sys.argv[1])
library = NaiReferenceLibrary(directory)
folder = library.add_folder("Folder")
reference = library.add(Image.create(Extent(8, 8), 0xFF0000FF), "First", folder_id=folder.id)
dialog = NaiReferenceDialog(model=SimpleNamespace(nai_references=library))
dialog.show()
app.processEvents()
assert dialog.items.topLevelItemCount() == 1
assert dialog.items.topLevelItem(0).childCount() == 1
dialog._move_item("reference", reference.id, None)
assert library.find(reference.id).folder_id is None

mime = QMimeData()
pixels = QImage(12, 12, QImage.Format.Format_RGBA8888)
pixels.fill(0xFFFF0000)
mime.setImageData(pixels)
enter = QDragEnterEvent(QPoint(3, 3), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
dialog.items.dragEnterEvent(enter)
assert enter.isAccepted()
drop = QDropEvent(QPointF(3, 3), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
dialog.items.dropEvent(drop)
assert drop.isAccepted()
assert len(list(library)) == 2
assert ReferenceDrop.from_mime(None) is None
internal = QMimeData()
internal.setData("application/x-krita-node-internal-pointer", b"node")
assert ReferenceDrop.from_mime(internal) is None
dialog.items.dropEvent(None)
dialog.dragEnterEvent(None)

menu = QMenu()
menu.setObjectName("drop_popup")
menu.addAction("Existing action")
drop_filter = NaiCanvasDropFilter(app)
drop_filter._pending = (ReferenceDrop.from_mime(mime), dialog)
assert not drop_filter.eventFilter(menu, QEvent(QEvent.Type.Show))
assert menu.actions()[0].objectName() == "ai_diffusion_insert_nai_reference"
dialog.reject()
app.processEvents()
restored = NaiReferenceLibrary(directory)
assert len(list(restored)) == 2
assert not restored.error
print("Offscreen reference dialog, image drop, folder movement, menu integration and reload passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONUTF8": "1"},
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
