import hashlib
from pathlib import Path

import pytest
from PyQt5.QtCore import pyqtBoundSignal

from ai_diffusion.model.updates import AutoUpdate, UpdateState, parse_release
from ai_diffusion.platform_tools import ZipFile

from .conftest import qtapp


class SignalObserver:
    def __init__(self, signal: pyqtBoundSignal):
        self.events = []
        signal.connect(self.on_changed)

    def on_changed(self, value):
        self.events.append(value)

    def reset(self):
        self.events = []


class FakeNetwork:
    """Stands in for RequestManager, serving a canned GitHub release."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.requests: list[str] = []

    async def get(self, url: str, timeout: float | None = None):
        self.requests.append(url)
        return self.responses[url]

    async def download(self, url: str):
        self.requests.append(url)
        return self.responses[url]


def release_json(tag: str, assets: list[str], base_url="https://example.com"):
    return {
        "tag_name": tag,
        "assets": [{"name": n, "browser_download_url": f"{base_url}/{n}"} for n in assets],
    }


def test_parse_release():
    zip_name = "krita_ai_diffusion-1.52.1-nai9.zip"
    package = parse_release(release_json("v1.52.1-nai9", [zip_name, f"{zip_name}.sha256"]))
    assert package is not None
    assert package.version == "1.52.1-nai9"  # leading "v" of the tag is stripped
    assert package.url == f"https://example.com/{zip_name}"
    assert package.checksum_url == f"https://example.com/{zip_name}.sha256"


def test_parse_release_without_checksum():
    zip_name = "krita_ai_diffusion-1.52.1-nai7.zip"
    package = parse_release(release_json("v1.52.1-nai7", [zip_name]))
    assert package is not None and package.checksum_url is None


@pytest.mark.parametrize(
    "release",
    [
        {},
        release_json("v1.52.1-nai9", []),  # release without any package attached
        release_json("v1.52.1-nai9", ["notes.txt"]),
        release_json("", ["krita_ai_diffusion-1.52.1-nai9.zip"]),
    ],
)
def test_parse_release_invalid(release):
    assert parse_release(release) is None


def build_package(build_dir: Path, name: str, content: str):
    """A minimal stand-in for what scripts/package_nai.py produces."""
    source = build_dir / "source" / "ai_diffusion"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(content)
    archive = build_dir / name
    with ZipFile(archive, "w") as zip_file:
        zip_file.write(source / "__init__.py", "ai_diffusion/__init__.py")
    return archive.read_bytes()


@qtapp
async def test_auto_update(tmp_path: Path):
    zip_name = "krita_ai_diffusion-1.52.1-nai9.zip"
    url = "https://example.com/" + zip_name
    archive = build_package(tmp_path / "build", zip_name, '__nai_version__ = "1.52.1-nai9"')
    net = FakeNetwork(
        {
            "https://api.github.com/repos/test/test/releases/latest": release_json(
                "v1.52.1-nai9", [zip_name, f"{zip_name}.sha256"]
            ),
            url: archive,
            f"{url}.sha256": f"{hashlib.sha256(archive).hexdigest()}  {zip_name}\n".encode(),
        }
    )

    install_dir = tmp_path / "install"
    installed = install_dir / "ai_diffusion" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text('__nai_version__ = "1.52.1-nai8"')

    updater = AutoUpdate(
        plugin_dir=install_dir,
        current_version="1.52.1-nai8",
        repository="test/test",
        net=net,  # type: ignore[arg-type]
    )
    assert updater.state is UpdateState.unknown

    state_changes = SignalObserver(updater.state_changed)
    await updater.check()
    assert state_changes.events == [UpdateState.checking, UpdateState.available]
    assert updater.latest_version == "1.52.1-nai9"
    assert updater.is_available

    state_changes.reset()
    await updater.run()
    assert state_changes.events == [
        UpdateState.downloading,
        UpdateState.installing,
        UpdateState.restart_required,
    ]
    assert installed.read_text() == '__nai_version__ = "1.52.1-nai9"'


@qtapp
async def test_auto_update_latest(tmp_path: Path):
    zip_name = "krita_ai_diffusion-1.52.1-nai9.zip"
    net = FakeNetwork(
        {
            "https://api.github.com/repos/test/test/releases/latest": release_json(
                "v1.52.1-nai9", [zip_name]
            )
        }
    )
    updater = AutoUpdate(tmp_path, "1.52.1-nai9", "test/test", net)  # type: ignore[arg-type]
    await updater.check()
    assert updater.state is UpdateState.latest
    assert not updater.is_available


@qtapp
async def test_auto_update_from_source(tmp_path: Path):
    """A source checkout has no release version and must not phone home."""
    net = FakeNetwork({})
    updater = AutoUpdate(tmp_path, "dev", "test/test", net)  # type: ignore[arg-type]
    await updater.check()
    assert updater.state is UpdateState.latest
    assert net.requests == []


@qtapp
async def test_auto_update_corrupt_package(tmp_path: Path):
    zip_name = "krita_ai_diffusion-1.52.1-nai9.zip"
    url = "https://example.com/" + zip_name
    archive = build_package(tmp_path / "build", zip_name, "corrupted")
    net = FakeNetwork(
        {
            "https://api.github.com/repos/test/test/releases/latest": release_json(
                "v1.52.1-nai9", [zip_name, f"{zip_name}.sha256"]
            ),
            url: archive,
            f"{url}.sha256": b"0" * 64 + b"  " + zip_name.encode(),
        }
    )
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    updater = AutoUpdate(install_dir, "1.52.1-nai8", "test/test", net)  # type: ignore[arg-type]
    await updater.check()
    await updater.run()
    assert updater.state is UpdateState.failed_update
    assert not (install_dir / "ai_diffusion").exists()  # nothing was installed
