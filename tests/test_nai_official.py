"""Golden-sample tests: does the plugin reproduce novelai.net field for field?

Every image novelai.net returns carries the server's echo of the request in its PNG
`Comment` chunk. tests/data/nai_official holds three of those, captured from real
renders. These tests rebuild the same request through the plugin's own builder and
compare, which is the only way to answer "did we send what the website sends?"
without spending Anlas on every run.

What they can and cannot prove: the Comment is a filtered echo, so it settles the
fields both sides carry and says nothing about the rest. The intersection is still
where the parameters that steer the sampler live, which is what matters at a fixed
seed. Pixel-level reproduction needs a live A/B run and is not testable here.
"""

import json
from pathlib import Path

import pytest

from ai_diffusion.backend.nai_request_builder import (
    ACTION_GENERATE,
    ACTION_IMG2IMG,
    NaiGenerationParams,
    build_request,
)
from ai_diffusion.image import Extent, Image

_DATA = Path(__file__).parent / "data" / "nai_official"
_MANIFEST = json.loads((_DATA / "manifest.json").read_text(encoding="utf-8"))

IGNORED_REQUEST_ONLY = frozenset(
    k for k in _MANIFEST["ignored_request_only"] if not k.startswith("_")
)
IGNORED_OFFICIAL_ONLY = frozenset(
    k for k in _MANIFEST["ignored_official_only"] if not k.startswith("_")
)
# Same rule one level down: the server fills in unsent keys inside the v4 caption
# blocks, so their presence in the echo is not evidence the client sent them.
IGNORED_NESTED_OFFICIAL_ONLY = frozenset(
    k for k in _MANIFEST["ignored_nested_official_only"] if not k.startswith("_")
)

# The website appends the transparency tag and the quality tags as one suffix
# (launcher api_constants.dart:607 composeSuffix), so peeling it back off is how
# the base prompt the user actually typed is recovered from a finished render.
_SUFFIX = {
    (True, True): ", transparent background, very aesthetic, masterpiece, no text",
    (True, False): ", very aesthetic, masterpiece, no text",
    (False, True): ", transparent background",
    (False, False): "",
}


def _samples():
    return [pytest.param(s, id=s["file"].removesuffix(".json")) for s in _MANIFEST["samples"]]


def _official(sample: dict) -> dict:
    return json.loads((_DATA / sample["file"]).read_text(encoding="utf-8"))


def _base_prompt(sample: dict, official: dict) -> str:
    suffix = _SUFFIX[(sample["quality_toggle"], sample["transparent_background"])]
    prompt = official["prompt"]
    if suffix:
        assert prompt.endswith(suffix), f"{sample['file']}: unexpected suffix in official prompt"
        prompt = prompt[: -len(suffix)]
    return prompt


def _plugin_request(sample: dict, official: dict, *, prompt: str | None = None) -> dict:
    """Rebuild the sample's request through the plugin's builder."""
    is_img2img = sample["action"] == "img2img"
    kwargs = {}
    if is_img2img:
        # Content does not matter for parameter comparison, only that a source
        # exists so the img2img branch runs.
        kwargs["source_image"] = Image.create(Extent(official["width"], official["height"]))
        kwargs["strength"] = official["strength"]
        kwargs["noise"] = official["noise"]

    params = NaiGenerationParams(
        model=sample["model"],
        action=ACTION_IMG2IMG if is_img2img else ACTION_GENERATE,
        prompt=official["prompt"] if prompt is None else prompt,
        negative_prompt=official["uc"],
        width=official["width"],
        height=official["height"],
        scale=official["scale"],
        steps=official["steps"],
        seed=official["seed"],
        sampler=official["sampler"],
        noise_schedule=official["noise_schedule"],
        cfg_rescale=official["cfg_rescale"],
        # The prompt is passed in already-assembled, so the suffix must not be
        # appended a second time. Prompt assembly has its own test below.
        quality_toggle=False,
        **kwargs,
    )
    return build_request(params).request_data


def _comparable(plugin: dict, official: dict) -> list[str]:
    shared = set(plugin) & set(official)
    return sorted(shared - IGNORED_REQUEST_ONLY - IGNORED_OFFICIAL_ONLY)


def _same(a, b) -> bool:
    # The server echoes whole floats as 1.0 where we send 1; JSON has one number type.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _describe(plugin, official, path: str) -> list[str]:
    """Report where two values diverge, not what they contain.

    v4_prompt carries the whole prompt twice over, so dumping mismatched values
    buries a missing boolean under two kilobytes of tags.
    """
    if isinstance(plugin, dict) and isinstance(official, dict):
        out = []
        for k in sorted(set(plugin) | set(official)):
            if k not in plugin:
                if f"{path}.{k}" in IGNORED_NESTED_OFFICIAL_ONLY:
                    continue  # server default, see the manifest for why
                out.append(f"{path}.{k}: missing from request (official: {official[k]!r})")
            elif k not in official:
                out.append(f"{path}.{k}: sent but not echoed ({plugin[k]!r})")
            else:
                out += _describe(plugin[k], official[k], f"{path}.{k}")
        return out
    if _same(plugin, official):
        return []
    if isinstance(plugin, str) and isinstance(official, str):
        for i, (x, y) in enumerate(zip(plugin, official)):
            if x != y:
                return [
                    f"{path}: diverges at char {i}: {plugin[i : i + 30]!r} vs {official[i : i + 30]!r}"
                ]
        longer = "request" if len(plugin) > len(official) else "official"
        extra = plugin[len(official) :] if len(plugin) > len(official) else official[len(plugin) :]
        return [f"{path}: {longer} has {len(extra)} extra trailing chars {extra!r}"]
    return [f"{path}: {plugin!r} vs {official!r}"]


@pytest.mark.parametrize("sample", _samples())
def test_request_matches_the_official_render_field_for_field(sample):
    """Every field both sides carry must be equal, not merely similar."""
    official = _official(sample)
    plugin = _plugin_request(sample, official)["parameters"]

    # tag_hint_qt is excluded here and covered by the prompt-assembly test: this
    # harness feeds the finished prompt with quality off, so the hint would report
    # the harness's setting rather than the sample's.
    fields = [f for f in _comparable(plugin, official) if f != "tag_hint_qt"]
    assert len(fields) >= 15, f"only {len(fields)} comparable fields - did the builder change?"

    problems = []
    for f in fields:
        problems += _describe(plugin[f], official[f], f)
    assert not problems, "differs from novelai.net:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("sample", _samples())
def test_sampling_parameters_are_actually_covered(sample):
    """Guard the guard.

    A field-for-field test passes trivially if the intersection is empty, so pin
    down that the fields which steer the sampler at a fixed seed are in it.
    """
    official = _official(sample)
    plugin = _plugin_request(sample, official)["parameters"]
    covered = set(_comparable(plugin, official))

    required = {
        "steps",
        "scale",
        "cfg_rescale",
        "seed",
        "sampler",
        "noise_schedule",
        "width",
        "height",
    }
    assert required <= covered, f"not compared: {sorted(required - covered)}"
    if sample["action"] == "img2img":
        assert {"strength", "noise", "extra_noise_seed"} <= covered


@pytest.mark.parametrize("sample", _samples())
def test_prompt_assembly_reproduces_the_official_prompt(sample):
    """The assembled prompt must match the website byte for byte.

    This is the sharp end of the comparison: the prompt is what the text encoder
    sees, so a single differing character is a different image at the same seed.
    """
    official = _official(sample)
    base = _base_prompt(sample, official)

    params = NaiGenerationParams(
        model=sample["model"],
        action=ACTION_IMG2IMG if sample["action"] == "img2img" else ACTION_GENERATE,
        prompt=base,
        negative_prompt=official["uc"],
        width=official["width"],
        height=official["height"],
        seed=official["seed"],
        quality_toggle=sample["quality_toggle"],
        quality_tier=sample["quality_tier"],
        transparent_background=sample["transparent_background"],
    )
    request = build_request(params).request_data

    assert request["input"] == official["prompt"]
    assert request["parameters"]["tag_hint_qt"] == official["tag_hint_qt"]
    assert request["parameters"]["tag_hint_uc_preset"] == official["tag_hint_uc_preset"]


def test_negative_prompt_reaches_the_server_under_the_name_it_echoes():
    """The plugin sends `negative_prompt`; every official render echoes `uc`.

    They carry the same text, so this pins the value rather than the key name -
    if the key were wrong the negative prompt would silently do nothing, which is
    invisible in output but obvious here.
    """
    sample = _MANIFEST["samples"][0]
    official = _official(sample)
    params = _plugin_request(sample, official)["parameters"]

    sent = params.get("negative_prompt", params.get("uc"))
    assert sent == official["uc"]
