"""Diff a NAI request the plugin sent against a render the website produced.

Reproducing novelai.net exactly means every field we send has to match what the
site sends. Both halves of that comparison need coaxing out: the site's request
survives only as the `Comment` tEXt chunk of the PNG it returned, and the
plugin's request is only on disk when `debug_dump_workflow` is on (see
`dump_nai_request` in ai_diffusion/backend/nai_client.py).

    python tools/nai_param_diff.py official.png nai-request-<id>.json

Development-time tool: runs outside Krita, so PIL/numpy are fair game (the
plugin runtime has neither).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

# Keys that legitimately appear on one side only. Each needs a reason, because
# an unexplained entry here is how a real difference gets hidden.
IGNORED_REQUEST_ONLY = {
    # The server echoes back a filtered subset in `Comment`; these are accepted
    # on the wire but never echoed. Verified empirically against three official
    # renders (1027-nai, 1027-nai-3, 图生图-nai): absent from all of them.
    "qualityPresetId",
    "ucPresetId",
    "image_format",
    # Payloads, compared as images rather than as fields.
    "image",
    "mask",
    "reference_image_multiple",
    "reference_information_extracted_multiple",
    "reference_strength_multiple",
    "director_reference_images",
    "director_reference_descriptions",
    "director_reference_information_extracted",
    "director_reference_strength_values",
}

IGNORED_OFFICIAL_ONLY = {
    # Stamped by the server, not by the client.
    "signed_hash",
    "request_type",
    "legacy_v3_extend",
}


def png_text_chunks(path: Path) -> dict[str, str]:
    data = path.read_bytes()[8:]
    out: dict[str, str] = {}
    i = 0
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        payload = data[i + 8 : i + 8 + length]
        i += 8 + length + 4
        if kind == b"tEXt":
            key, _, value = payload.partition(b"\x00")
            out[key.decode("latin1")] = value.decode("utf-8", "replace")
        elif kind == b"IEND":
            break
    return out


def official_parameters(path: Path) -> dict:
    chunks = png_text_chunks(path)
    if "Comment" not in chunks:
        raise SystemExit(f"{path} has no NAI `Comment` chunk — is it a website render?")
    params = json.loads(chunks["Comment"])
    # The website mirrors the prompt into `Comment` too; keep it, it is the one
    # field where prompt assembly shows up.
    return params


def plugin_parameters(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    params = dict(request.get("parameters", {}))
    # `action` and `input` live at the top level of the request but inside
    # `Comment` on the official side, so lift them to make the shapes comparable.
    for key in ("action", "input", "model"):
        if key in request:
            params.setdefault(key, request[key])
    return params


def normalize(value):
    """1 and 1.0 are the same number on the wire; don't report that as a diff."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    official = official_parameters(Path(argv[1]))
    plugin = plugin_parameters(Path(argv[2]))

    # The official `Comment` carries the prompt under `prompt`; our request calls
    # it `input`. Same field, different name on each side.
    if "prompt" in official and "input" in plugin:
        plugin["prompt"] = plugin.pop("input")

    differing = sorted(
        k for k in official.keys() & plugin.keys() if normalize(official[k]) != normalize(plugin[k])
    )
    request_only = sorted(
        k for k in plugin.keys() - official.keys() if k not in IGNORED_REQUEST_ONLY
    )
    official_only = sorted(
        k for k in official.keys() - plugin.keys() if k not in IGNORED_OFFICIAL_ONLY
    )

    def show(value) -> str:
        text = json.dumps(value, ensure_ascii=False)
        return text if len(text) <= 160 else text[:157] + "..."

    print(f"=== 值不同 ({len(differing)}) ===")
    for key in differing:
        print(f"  {key}")
        print(f"    官网: {show(official[key])}")
        print(f"    插件: {show(plugin[key])}")
    print(f"\n=== 仅请求有 ({len(request_only)}) ===")
    for key in request_only:
        print(f"  {key} = {show(plugin[key])}")
    print(f"\n=== 仅官网有 ({len(official_only)}) ===")
    for key in official_only:
        print(f"  {key} = {show(official[key])}")

    total = len(differing) + len(request_only) + len(official_only)
    print(f"\n{total} 处差异" if total else "\n无差异 — 参数已复刻")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
