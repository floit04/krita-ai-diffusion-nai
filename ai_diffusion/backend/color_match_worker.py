"""Local color matching worker; no network or Krita imports.

ColorMatcher() intentionally keeps the color-matcher 0.6.0 default solver.
Histogram GPU port based on Christopher Hahne's color-matcher hist_matcher.py.
Copyright (c) 2020 Christopher Hahne. Modifications (c) 2026.
SPDX-License-Identifier: GPL-3.0-or-later
https://github.com/hahnec/color-matcher
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
from color_matcher import ColorMatcher  # pyright: ignore[reportMissingImports]
from PIL import Image

METHOD = "hm-mvgd-hm"


def gpu_hist_match(src, ref):
    import cupy as cp  # pyright: ignore[reportMissingImports]

    s, r = cp.asarray(src), cp.asarray(ref)
    out = cp.zeros_like(s)
    for ch in range(s.shape[2]):
        sv, rv = s[..., ch].ravel(), r[..., ch].ravel()
        _, indexes, counts = cp.unique(sv, return_inverse=True, return_counts=True)
        values, ref_counts = cp.unique(rv, return_counts=True)
        cdf = cp.cumsum(counts).astype(cp.float64) / sv.size
        ref_cdf = cp.cumsum(ref_counts).astype(cp.float64) / rv.size
        out[..., ch] = cp.interp(cdf, ref_cdf, values)[indexes].reshape(s[..., ch].shape)
    return cp.asnumpy(out)


def transfer(src, ref, backend="cpu"):
    matcher = ColorMatcher()
    if backend == "gpu":
        matcher.hist_match = gpu_hist_match
    result = matcher.transfer(src=src, ref=ref, method=METHOD)
    if np.iscomplexobj(result):
        if float(np.max(np.abs(result.imag))) > 1e-7:
            raise ValueError("Unstable covariance transfer; preserve original")
        result = result.real
    if not np.isfinite(result).all():
        raise ValueError("Non-finite color matching result")
    return np.clip(result, 0, 1)


def configured_backend(pixels, config_path):
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for row in sorted(config.get("backend_ranges", []), key=lambda x: x["max_pixels"]):
            if pixels <= row["max_pixels"]:
                return row["backend"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return "cpu"


def match_file(src_path, ref_path, out_path, backend="cpu"):
    started = time.perf_counter()
    with Image.open(src_path) as im:
        src_rgba = np.array(im.convert("RGBA"))
        profile = im.info.get("icc_profile")
    with Image.open(ref_path) as im:
        ref_rgba = np.array(im.convert("RGBA"))
    if src_rgba.shape != ref_rgba.shape:
        raise ValueError("Reference and generated image extents differ")
    src = src_rgba[..., :3].astype(np.float32) / 255.0
    ref = ref_rgba[..., :3].astype(np.float32) / 255.0
    valid = (src_rgba[..., 3] > 0) & (ref_rgba[..., 3] > 0)
    if int(valid.sum()) < 16:
        raise ValueError("Too few visible reference pixels")
    all_visible = bool(valid.all())
    s = src if all_visible else src[valid].reshape(-1, 1, 3)
    r = ref if all_visible else ref[valid].reshape(-1, 1, 3)
    if np.any(r.reshape(-1, 3).std(axis=0) < 1e-6):
        raise ValueError("Uniform reference color channel; preserve original")
    if np.any(s.reshape(-1, 3).std(axis=0) < 1e-6):
        raise ValueError("Uniform generated color channel; preserve original")
    actual_backend = backend
    gpu_error = None
    try:
        matched = transfer(s, r, backend)
    except Exception as error:
        if backend != "gpu":
            raise
        gpu_error = str(error)
        actual_backend = "cpu"
        matched = transfer(s, r, "cpu")
    rgb = np.rint(matched * 255).astype(np.uint8)
    result = src_rgba.copy()
    if all_visible:
        result[..., :3] = rgb
    else:
        result[..., :3][valid] = rgb.reshape(-1, 3)
    kwargs: dict = {"compress_level": 1}
    if profile:
        kwargs["icc_profile"] = profile
    Image.fromarray(result).save(out_path, **kwargs)
    return {
        "ok": True,
        "backend": actual_backend,
        "gpu_error": gpu_error,
        "seconds": time.perf_counter() - started,
    }


def run_manifest(path, backend=None):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    report = []
    for item in manifest["items"]:
        try:
            with Image.open(item["source"]) as im:
                pixels = im.width * im.height
            selected = backend or configured_backend(pixels, manifest["config"])
            row = match_file(item["source"], manifest["reference"], item["output"], selected)
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        report.append(row)
    Path(manifest["report"]).write_text(json.dumps(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--backend", choices=["cpu", "gpu"])
    args = parser.parse_args()
    print(json.dumps(run_manifest(args.manifest, args.backend)))
