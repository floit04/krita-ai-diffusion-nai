import hashlib
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from PyQt5.QtCore import QObject, pyqtSignal

from .. import __nai_version__, eventloop
from ..backend.network import RequestManager
from ..platform_tools import ZipFile
from ..util import client_logger as log
from .properties import ObservableProperties, Property


class UpdateState(Enum):
    unknown = 1
    checking = 2
    available = 3
    latest = 4
    downloading = 5
    installing = 6
    restart_required = 7
    failed_check = 8
    failed_update = 9


class UpdatePackage(NamedTuple):
    version: str
    url: str
    checksum_url: str | None = None


def parse_release(release: dict) -> UpdatePackage | None:
    """Pick the plugin package out of a GitHub release, or None if there is none.

    Releases are built by .github/workflows/release-nai.yml, which uploads the
    importable plugin zip plus a `<zip>.sha256` sidecar next to it.
    """
    tag = release.get("tag_name") or ""
    version = tag[1:] if tag.startswith("v") else tag
    assets = release.get("assets") or []
    names = {a.get("name"): a.get("browser_download_url") for a in assets}
    zip_name = next((n for n in names if n and n.endswith(".zip")), None)
    if not version or not zip_name or not names[zip_name]:
        return None
    return UpdatePackage(version, names[zip_name], names.get(f"{zip_name}.sha256"))


class AutoUpdate(QObject, ObservableProperties):
    # Updates come from this fork's own releases, not from upstream: whoever installed
    # the NovelAI edition wants NovelAI edition updates. Syncing with upstream is a
    # manual step which produces a new release here.
    default_repository = os.getenv("KRITA_AI_NAI_REPOSITORY", "floit04/krita-ai-diffusion-nai")

    state = Property(UpdateState.unknown)
    latest_version = Property("")
    error = Property("")

    state_changed = pyqtSignal(UpdateState)
    latest_version_changed = pyqtSignal(str)
    error_changed = pyqtSignal(str)

    def __init__(
        self,
        plugin_dir: Path | None = None,
        current_version: str | None = None,
        repository: str | None = None,
        net: RequestManager | None = None,
    ):
        super().__init__()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent.parent
        self.current_version = current_version or __nai_version__
        self.repository = repository or self.default_repository
        self._package: UpdatePackage | None = None
        self._temp_dir: TemporaryDirectory | None = None
        self._request_manager: RequestManager | None = net

    def check(self):
        return eventloop.run(
            self._handle_errors(
                self._check, UpdateState.failed_check, "Failed to check for new plugin version"
            )
        )

    async def _check(self):
        if self.state is UpdateState.restart_required:
            return
        if self.current_version == "dev":
            # A source checkout, not an installed release package: there is nothing to
            # compare against, and overwriting it with a zip would be wrong.
            log.info("Plugin runs from source, skipping update check")
            self.latest_version = self.current_version
            self.state = UpdateState.latest
            return

        self.state = UpdateState.checking
        url = f"https://api.github.com/repos/{self.repository}/releases/latest"
        log.info(f"Checking for latest plugin version at {url}")
        result = await self._net.get(url, timeout=10)
        package = parse_release(result) if isinstance(result, dict) else None
        if package is None:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Failed to retrieve plugin update information"
            return

        self._package = package
        self.latest_version = package.version
        if package.version == self.current_version:
            log.info("Plugin is up to date!")
            self.state = UpdateState.latest
        else:
            log.info(f"New plugin version available: {self.latest_version}")
            self.state = UpdateState.available

    def run(self):
        return eventloop.run(
            self._handle_errors(self._run, UpdateState.failed_update, "Failed to update plugin")
        )

    async def _run(self):
        assert self.latest_version and self._package

        self._temp_dir = TemporaryDirectory()
        archive_path = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}.zip"
        log.info(f"Downloading plugin update {self._package.url}")
        self.state = UpdateState.downloading
        archive_data = await self._net.download(self._package.url)
        await self._verify_checksum(archive_data)

        archive_path.write_bytes(archive_data)
        source_dir = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}"
        log.info(f"Extracting plugin archive into {source_dir}")
        self.state = UpdateState.installing
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(source_dir)
        if not (source_dir / "ai_diffusion" / "__init__.py").exists():
            # The archive is copied over a working installation, so make sure it really
            # is a plugin package before touching anything.
            raise RuntimeError("Downloaded package does not contain a plugin")

        log.info(f"Installing new plugin version to {self.plugin_dir}")
        shutil.copytree(source_dir, self.plugin_dir, dirs_exist_ok=True)
        self.current_version = self.latest_version
        self.state = UpdateState.restart_required

    @property
    def is_available(self):
        return self.latest_version is not None and self.latest_version != self.current_version

    async def _verify_checksum(self, archive_data: bytes):
        assert self._package
        if not self._package.checksum_url:
            # Releases published before the sidecar was introduced don't have one.
            log.warning("Release has no .sha256 asset, skipping package checksum")
            return

        checksum = await self._net.get(self._package.checksum_url, timeout=10)
        if isinstance(checksum, bytes):
            checksum = checksum.decode("utf-8", errors="replace")
        expected = str(checksum).split()[0].lower()  # "<hash>  <filename>"
        sha256 = hashlib.sha256(archive_data).hexdigest()
        if sha256 != expected:
            log.error(f"Update package hash mismatch: {sha256} != {expected}")
            raise RuntimeError("Downloaded plugin package is corrupted or incomplete")

    @property
    def _net(self):
        if self._request_manager is None:
            self._request_manager = RequestManager()
            # The GitHub API rejects requests without a User-Agent. No Accept header:
            # the API defaults to application/json, which is what the response parser
            # needs to see in order to decode it.
            self._request_manager.add_header("User-Agent", "krita-ai-diffusion-nai")
        return self._request_manager

    async def _handle_errors(self, func, error_state: UpdateState, message: str):
        try:
            return await func()
        except Exception as e:
            log.exception(e)
            self.error = f"{message}: {e}"
            self.state = error_state
            return None
