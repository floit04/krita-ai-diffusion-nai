"""Generative AI plugin for Krita"""

__version__ = "1.52.1"

# Version of this NovelAI fork. Stamped by scripts/package_nai.py when a release
# package is built (e.g. "1.52.1-nai8"); "dev" means the plugin runs from a source
# checkout, in which case there is no release to compare against and the
# auto-updater stays quiet. Kept separate from __version__, which identifies the
# upstream base version and is what the cloud backend reports.
__nai_version__ = "dev"

# What the UI shows. Source checkouts have no release version of their own.
__display_version__ = __nai_version__ if __nai_version__ != "dev" else f"{__version__}-dev"

import importlib.util

if not importlib.util.find_spec(".websockets.src", "ai_diffusion"):
    raise ImportError(
        "Could not find websockets module. This indicates that it was not installed with the"
        " plugin. Please make sure to download a plugin release package (NOT just the source!). You"
        " can find the latest release package here:"
        " https://github.com/floit04/krita-ai-diffusion-nai/releases"
    )

# The following imports depend on the code running inside Krita
if importlib.util.find_spec("krita"):
    import krita

    if not getattr(krita, "IS_MOCK", False):
        krita_ver = krita.Krita.instance().version()
        if not krita_ver.startswith("5"):
            raise ImportError(f"This Plugin is for Krita 5.x, but you are using Krita {krita_ver}.")

        from .extension import AIToolsExtension as AIToolsExtension

# When not running inside Krita, try to import the development placeholder for Krita functions
else:
    import sys
    from pathlib import Path

    mock_dir = Path(__file__).parent.parent / "tests" / "mock"
    if mock_dir.exists():
        sys.path.append(str(mock_dir))
