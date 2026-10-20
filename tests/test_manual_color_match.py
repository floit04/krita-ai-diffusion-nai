from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, QPoint, QRect, QUuid

from ai_diffusion.image import Bounds
from ai_diffusion.model.manual_color_match import (
    MATCH_LAYER_SUFFIX,
    ManualColorMatch,
    MatchLayer,
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


def test_match_layer_is_looked_up_by_id():
    model = Model()
    layer = object()
    match_id = QUuid.createUuid()
    finder = SimpleNamespace(find=lambda i: layer if i == match_id else None)
    finder.updated = lambda: finder
    model.document = SimpleNamespace(layers=finder)
    controller = ManualColorMatch(model)
    controller._timer.stop()

    node = Node()
    controller._states[node.uniqueId().toString()] = MatchLayer(match_id)

    assert controller._match_layer(model, node) is layer
    assert controller._match_layer(model, Node()) is None


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


def test_add_match_layer_clones_the_source_and_sets_the_pixels():
    model = Model()
    controller = ManualColorMatch(model)
    controller._timer.stop()

    parent = CloneNode("root")
    source = CloneNode("ink", parent)
    parent.children.append(source)
    wrapper = SimpleNamespace(layers=SimpleNamespace(update=lambda: None))
    doc = Doc()

    output = bytes([1, 2, 3, 255])
    layer = controller._add_match_layer(wrapper, doc, source, Bounds(0, 0, 1, 1), output)

    # Inserted directly above the source, named after it, carrying the new pixels.
    assert parent.children == [source, layer]
    assert layer.name() == "ink" + MATCH_LAYER_SUFFIX
    assert layer.data == output
    # The source is untouched, which is what makes Ctrl+Z a pure layer removal.
    assert source.data == bytes([20, 30, 40, 255])
