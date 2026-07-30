"""Build the importable Krita plugin zip for the NAI edition.

Usage: python scripts/package_nai.py [suffix]
  suffix: optional version suffix, e.g. "nai4" -> krita_ai_diffusion-<version>-nai4.zip
          defaults to "nai".

The zip root contains ai_diffusion.desktop + ai_diffusion/ so it can be imported
directly via Krita's "Import Python Plugin from File". Mirrors the upstream
scripts/package.py layout rules (no hidden files, no __pycache__, no debugpy).
No third-party dependencies — safe to run in CI.
"""

import re
import sys
from pathlib import Path
from shutil import copy, copytree, ignore_patterns, make_archive, rmtree

root = Path(__file__).parent.parent


def plugin_version() -> str:
    text = (root / "ai_diffusion" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([^"]+)"', text)
    assert match, "could not find __version__ in ai_diffusion/__init__.py"
    return match.group(1)


def build(suffix: str = "nai") -> Path:
    name = f"krita_ai_diffusion-{plugin_version()}-{suffix}"
    package_dir = root / "scripts" / ".package"
    rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    copy(root / "ai_diffusion.desktop", package_dir)

    def ignore(path, names):
        return ignore_patterns(".*", "*.pyc", "__pycache__", "debugpy")(path, names)

    copytree(root / "ai_diffusion", package_dir / "ai_diffusion", ignore=ignore)
    copy(root / "LICENSE", package_dir / "ai_diffusion")
    copy(root / "README_NAI.md", package_dir / "ai_diffusion")

    archive = Path(make_archive(str(root / name), "zip", package_dir))
    rmtree(package_dir)
    return archive


if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else "nai"
    result = build(suffix)
    print(result)
