"""Asynchronous, isolated local color matching. No network access."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..image import Image, ImageCollection
from ..util import client_logger as log


async def match_nai_images(images, reference_base64, cancelled=lambda: False):
    process = None
    try:
        home = Path(os.environ["LOCALAPPDATA"]) / "KritaColorMatch"
        config_path = home / "runtime.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        python = Path(config["python"])
        worker = Path(__file__).with_name("color_match_worker.py")
        if not python.is_file() or not worker.is_file():
            raise FileNotFoundError("Local color matching runtime is missing")
        with tempfile.TemporaryDirectory(prefix="krita-colormatch-") as folder:
            temp = Path(folder)
            reference = temp / "reference.png"
            reference.write_bytes(base64.b64decode(reference_base64, validate=True))
            items = []
            for i, image in enumerate(images):
                source, output = temp / f"source-{i}.png", temp / f"output-{i}.png"
                source.write_bytes(bytes(image.to_bytes()))
                items.append({"source": str(source), "output": str(output)})
            report_path = temp / "report.json"
            manifest = {
                "reference": str(reference),
                "config": str(config_path),
                "items": items,
                "report": str(report_path),
            }
            manifest_path = temp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (temp / "worker.log").open("wb") as output_log:
                env = os.environ.copy()
                for key in ("PYTHONHOME", "PYTHONPATH"):
                    env.pop(key, None)
                env["PYTHONUTF8"] = "1"
                started = time.monotonic()
                # Krita's Qt event loop lacks asyncio subprocess transport support.
                # Poll asynchronously below; stdout/stderr go to disk, never PIPE.
                process = subprocess.Popen(  # noqa: ASYNC220
                    [str(python), "-I", str(worker), str(manifest_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=output_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(home),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                try:
                    while process.poll() is None:
                        if cancelled():
                            raise asyncio.CancelledError()
                        if time.monotonic() - started > 120:
                            raise TimeoutError("Local color matching exceeded 120 seconds")
                        await asyncio.sleep(0.05)
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"Color matching worker exited with code {process.returncode}"
                        )
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
            if cancelled():
                raise asyncio.CancelledError()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if len(report) != len(items):
                raise ValueError("Incomplete color matching report")
            matched = ImageCollection()
            for original, item, status in zip(images, items, report):
                if status.get("ok"):
                    try:
                        corrected = Image.load(item["output"])
                        if corrected.extent != original.extent:
                            raise ValueError("Color matching changed image dimensions")
                        matched.append(corrected)
                        log.info(
                            "Local layer color match: HM-MVGD-HM, %s, %.3fs",
                            status.get("backend"),
                            status.get("seconds", 0),
                        )
                        if status.get("gpu_error"):
                            log.warning(
                                "Local color match GPU failed; CPU used: %s", status["gpu_error"]
                            )
                        continue
                    except Exception:
                        log.exception("Cannot load color-matched image; keeping original")
                else:
                    log.warning(
                        "Local color match skipped; keeping original: %s", status.get("error")
                    )
                matched.append(original)
            return matched
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Local layer color match failed; keeping original images")
        return images
