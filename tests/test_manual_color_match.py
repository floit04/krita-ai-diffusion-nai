from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, QPoint, QRect, QUuid

from ai_diffusion.image import Bounds
from ai_diffusion.model.manual_color_match import (
    MATCH_LAYER_PREFIX,
    ManualColorMatch,
    MatchSnapshot,
    eligibility,
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
    def __init__(self):
        self.active = None

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

    def setActiveNode(self, node):
        self.active = node


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


def test_match_snapshot_is_looked_up_by_node_uid():
    model = Model()
    controller = ManualColorMatch(model)
    controller._timer.stop()

    node = Node()
    snap = MatchSnapshot(Bounds(0, 0, 1, 1), node.data, "ink")
    controller._states[node.uniqueId().toString()] = snap

    assert controller._match_snapshot(node) is snap
    assert controller._match_snapshot(Node()) is None


class CloneNode:
    """Just enough of a krita.Node for the match layer insertion."""

    def __init__(self, name="ink", parent=None, data=bytes([20, 30, 40, 255])):
        self._name = name
        self._parent = parent
        self.data = data
        self.children: list[CloneNode] = []

    def name(self):
        return self._name

    def setName(self, value):
        self._name = value

    def duplicate(self):
        copy = CloneNode(self._name, self._parent, self.data)
        return copy

    def setPixelData(self, data, *args):
        self.data = bytes(data)
        return True

    def parentNode(self):
        return self._parent

    def addChildNode(self, child, above):
        child._parent = self
        self.children.insert(self.children.index(above) + 1, child)
        return True

    def remove(self):
        if self._parent is not None:
            self._parent.children.remove(self)
            return True
        return False


def test_replace_layer_replaces_source_with_prefixed_clone_and_restores():
    model = Model()
    controller = ManualColorMatch(model)
    controller._timer.stop()

    parent = CloneNode("root")
    source = CloneNode("ink", parent)
    parent.children.append(source)
    wrapper = SimpleNamespace(layers=SimpleNamespace(update=lambda: None))
    doc = Doc()

    output = bytes([1, 2, 3, 255])
    layer = controller._replace_layer(
        wrapper, doc, source, Bounds(0, 0, 1, 1), output, f"{MATCH_LAYER_PREFIX}ink"
    )

    assert parent.children == [layer]
    assert layer.name() == f"{MATCH_LAYER_PREFIX}ink"
    assert layer.data == output
    assert doc.active is layer
    assert source.data == bytes([20, 30, 40, 255])

    restored = controller._replace_layer(
        wrapper, doc, layer, Bounds(0, 0, 1, 1), source.data, "ink"
    )

    assert parent.children == [restored]
    assert restored.name() == "ink"
    assert restored.data == source.data
    assert doc.active is restored
