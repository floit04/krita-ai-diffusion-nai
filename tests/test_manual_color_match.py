from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, QPoint, QRect, QUuid

from ai_diffusion.image import Bounds
from ai_diffusion.model.manual_color_match import (
    ManualColorMatch,
    RestorePoint,
    digest,
    eligibility,
    geometry,
    path_to,
)


class Model(QObject):
    def __init__(self):
        super().__init__()
        self.document = SimpleNamespace(is_valid=False)
        self.errors = []

    def report_error(self, text):
        self.errors.append(text)


class Node:
    def __init__(self):
        self.id = QUuid.createUuid()
        self.data = bytes([20, 30, 40, 255])
        self.pos = QPoint(0, 0)
        self.hidden = False
        self.lock = False

    def uniqueId(self):
        return self.id

    def bounds(self):
        return QRect(0, 0, 1, 1)

    def position(self):
        return self.pos

    def colorModel(self):
        return "RGBA"

    def colorDepth(self):
        return "U8"

    def colorProfile(self):
        return "sRGB"

    def type(self):
        return "paintlayer"

    def animated(self):
        return False

    def locked(self):
        return self.lock

    def visible(self):
        return not self.hidden

    def childNodes(self):
        return []

    def parentNode(self):
        return None

    def pixelData(self, *args):
        return self.data

    def setPixelData(self, data, *args):
        self.data = bytes(data)
        return True


class Doc:
    def width(self):
        return 1

    def height(self):
        return 1

    def currentTime(self):
        return 0

    def colorModel(self):
        return "RGBA"

    def colorDepth(self):
        return "U8"

    def colorProfile(self):
        return "sRGB"

    def waitForDone(self):
        pass

    def refreshProjection(self):
        pass

    def setModified(self, value):
        pass


def test_initialization_without_handoff_placeholder():
    model = Model()
    controller = ManualColorMatch(model)
    assert controller._states == {}
    assert controller.status()[:2] == (False, False)
    controller._timer.stop()


@pytest.mark.parametrize("flag", ["lock", "hidden"])
def test_ineligible_layer(flag):
    node = Node()
    setattr(node, flag, True)
    assert eligibility(Doc(), node)


def test_supported_layer_and_path():
    node = Node()
    assert eligibility(Doc(), node) == ""
    assert path_to(node, node.id.toString()) == ()
    assert path_to(node, "missing") is None


@pytest.mark.parametrize("change", ["none", "pixels", "position", "backup"])
def test_restore_protects_later_changes(tmp_path: Path, change):
    model, node, doc = Model(), Node(), Doc()
    controller = ManualColorMatch(model)
    controller._timer.stop()
    original = bytes([1, 2, 3, 255])
    path = tmp_path / "original.bgra"
    path.write_bytes(original)
    point = RestorePoint(
        Bounds(0, 0, 1, 1), geometry(node), path, digest(original), digest(node.data), (1, 1, 0)
    )
    controller._states[node.id.toString()] = point
    if change == "pixels":
        node.data = bytes([9, 9, 9, 255])
    elif change == "position":
        node.pos = QPoint(1, 0)
    elif change == "backup":
        path.write_bytes(b"corrupt")
    before = node.data
    if change == "none":
        controller._restore(doc, node, point)
        assert node.data == original
        assert not controller._states
    else:
        with pytest.raises(RuntimeError):
            controller._restore(doc, node, point)
        assert node.data == before
