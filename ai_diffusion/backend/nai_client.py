"""NovelAI Image Generation API client.

Implements the Client ABC to integrate NovelAI as a generation backend.
Communication is done via HTTP POST requests; NAI does not provide
WebSocket-based progress streaming, so the client reports a simple
"generating" / "finished" two-state progress.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt5.QtNetwork import QNetworkReply

from .. import util
from ..image import Bounds, Extent, Image, ImageCollection
from ..localization import translate as _
from ..settings import PerformanceSettings, settings
from ..util import client_logger as log
from ..util import ensure
from .api import WorkflowInput
from .client import (
    Client,
    ClientEvent,
    ClientFeatures,
    ClientJobQueue,
    ClientMessage,
    ClientModels,
    DeviceInfo,
)
from .nai_mask_utils import InpaintMaskArtifacts, apply_composite_mask
from .nai_registry import QualityTags, UcPresets
from .nai_request_builder import (
    ACTION_GENERATE,
    ACTION_IMG2IMG,
    ACTION_INFILL,
    NaiGenerationParams,
    PreciseReference,
    VibeReference,
    build_request,
)
from .nai_workflow import (
    NAI_EDIT_MAX_PIXELS,
    NAI_MAX_PIXELS,
    NaiAction,
    NaiModel,
    NaiNoiseSchedule,
    NaiSampler,
    NaiUCPreset,
    clamp_resolution,
    image_to_base64,
    map_noise_schedule,
    map_sampler,
    nai_auto_resolution,
    prepare_nai_precise_reference_image,
)
from .network import NetworkError, RequestManager

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------


class NaiJobState(Enum):
    pending = 1
    generating = 2
    completed = 3
    failed = 4
    cancelled = 5


@dataclass
class NaiJobInfo:
    local_id: str
    work: WorkflowInput
    state: NaiJobState = NaiJobState.pending

    # NAI-specific generation parameters (built by convert_workflow)
    nai_request: dict[str, Any] | None = None

    def __str__(self):
        return f"NaiJob[{self.work.kind.name}, id={self.local_id}]"


# ---------------------------------------------------------------------------
# Vibe encoding disk cache (ai/encode-vibe costs 2 Anlas per image — never pay
# twice for the same image, even across Krita restarts)
# ---------------------------------------------------------------------------


def _vibe_cache_path():
    from ..util import user_data_dir

    return user_data_dir / "nai_vibe_cache.json"


def _load_vibe_cache() -> dict[str, str]:
    try:
        path = _vibe_cache_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        log.warning(f"NAI: failed to load vibe cache: {e}")
    return {}


def _save_vibe_cache(cache: dict[str, str]):
    try:
        _vibe_cache_path().write_text(json.dumps(cache), encoding="utf-8")
    except Exception as e:
        log.warning(f"NAI: failed to save vibe cache: {e}")


# ---------------------------------------------------------------------------
# E-mail + password login
# ---------------------------------------------------------------------------

# User endpoints live on the IMAGE host for third-party clients — api.novelai.net
# answers 400 "update to the image URL". (Launcher: nai_api_endpoint.dart userUrl)
nai_user_api_url = "https://image.novelai.net"


async def login_with_password(email: str, password: str) -> str:
    """Log in with NovelAI credentials and return the bearer token.

    The password never leaves this machine: an access key is derived from it with
    Argon2id and only that key is sent to ``/user/login``. Key derivation takes
    about a second of pure-Python number crunching, so it runs in a worker thread
    to keep Krita responsive.

    Returns the JWT access token (valid ~30 days), which is used exactly like a
    persistent ``pst-`` token.
    """
    from . import nai_auth

    loop = asyncio.get_event_loop()
    access_key = await loop.run_in_executor(None, nai_auth.derive_access_key, email, password)

    requests = RequestManager()
    try:
        data = await requests.post(f"{nai_user_api_url}/user/login", {"key": access_key})
    except NetworkError as e:
        if e.status == 401:  # server answers "Invalid access key"
            raise RuntimeError(_("Wrong e-mail or password.")) from e
        raise
    token = data.get("accessToken") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError(_("NovelAI login did not return an access token"))
    return str(token)


# ---------------------------------------------------------------------------
# NAI Client
# ---------------------------------------------------------------------------


class NaiClient(Client):
    """NovelAI Image Generation API client.

    Authenticates with a persistent API token (``pst-xxx``) and sends
    generation requests to ``https://image.novelai.net``.
    """

    default_api_url = "https://image.novelai.net"

    async def connect(self):
        if not self._token:
            raise ValueError("NovelAI API token is required")
        # Validate the token with a lightweight request (tag suggestion)
        try:
            await self._validate_token()
        except NetworkError as e:
            if e.status == 401:
                e.message = _("NovelAI API token is invalid or expired. Please check your token.")
            raise
        log.info(f"Connected to NovelAI API at {self.url}")

    def __init__(self, url: str, token: str = ""):
        self.url = url.rstrip("/")
        self.models = _build_nai_models()
        self.device_info = DeviceInfo("Cloud", "NovelAI", 0)
        self._requests = RequestManager()
        self._token = token.strip()
        self._queue: ClientJobQueue[NaiJobInfo] = ClientJobQueue()
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._is_connected = False
        self._runner_task: asyncio.Task | None = None
        self._current_job: NaiJobInfo | None = None
        self._features = ClientFeatures(
            ip_adapter=False,
            translation=False,
            languages=[],
            max_upload_size=0,
            # NAI control layers: img2img base / vibe transfer / precise reference.
            max_control_layers=16,
        )
        # Vibe encodings cost 2 Anlas each (ai/encode-vibe) — cache per image/model,
        # persisted to disk so an image is never paid for twice across restarts.
        self._vibe_cache: dict[str, str] = _load_vibe_cache()

    def _current_token(self) -> str:
        """Always use the CURRENT token from settings.

        The token stored at connect time goes stale when the user switches the
        active token in settings — requests would then hit the wrong account
        (symptom: "Not enough Anlas" despite a topped-up account).
        """
        return (settings.nai_api_token or self._token).strip()

    # -- HTTP helpers -------------------------------------------------------

    async def _get(self, path: str, timeout: float | None = 30):
        return await self._requests.get(
            f"{self.url}/{path}", timeout=timeout, bearer=self._current_token()
        )

    async def _post(self, path: str, data: dict, timeout: float | None = None):
        return await self._requests.post(f"{self.url}/{path}", data, bearer=self._current_token())

    async def _post_binary(self, path: str, data: dict, timeout: float | None = 300):
        """POST that expects a binary (ZIP) response.

        NAI occasionally drops the connection mid-download (RemoteHostClosedError,
        code=2) or times out. Retry a few times like network.download() does before
        surfacing the error, so a transient hiccup doesn't fail the whole job.

        Requests that never got an HTTP response at all (``status is None``) are
        also retried: the body upload failed before the server answered, so no
        image was generated and no Anlas was spent — retrying cannot double-charge.
        This covers restrictive networks that kill multi-MB inpaint uploads
        (socket "Unable to write", connection reset by a corporate gateway, ...).
        An HTTP error such as 500 is NOT retried — the server may already have
        generated (and charged for) the image.
        """
        for retry in range(3, 0, -1):
            try:
                return await self._requests.http(
                    "POST",
                    f"{self.url}/{path}",
                    data,
                    timeout=timeout,
                    bearer=self._current_token(),
                )
            except NetworkError as e:
                transient = e.status is None or e.code in (
                    QNetworkReply.NetworkError.RemoteHostClosedError,
                    QNetworkReply.NetworkError.TemporaryNetworkFailureError,
                    QNetworkReply.NetworkError.TimeoutError,
                )
                if not transient or retry == 1:
                    raise
                log.warning(f"NAI request interrupted ({e}); retrying, {retry - 1} left")
                await asyncio.sleep(1)

    async def _validate_token(self):
        """Lightweight request to verify that the API token is valid."""
        await self._get(
            "ai/generate-image/suggest-tags?model=nai-diffusion-3&prompt=test",
            timeout=15,
        )

    # -- Client ABC implementation ------------------------------------------

    async def discover_models(self, refresh: bool):
        # NAI models are a fixed, built-in set (see _build_nai_models); there is
        # nothing to scan on the server. Yield a single completed status so the
        # connection flow's `async for` progresses.
        self.models = _build_nai_models()
        n = len(self.models.checkpoints)
        yield self.DiscoverStatus(folder="models", current=n, total=n)

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        job = NaiJobInfo(str(uuid.uuid4()), work)
        self._queue.put(job, front=front)
        return job.local_id

    async def listen(self):
        assert not self._is_connected, "NaiClient is already connected"
        self._is_connected = True
        self._runner_task = asyncio.get_running_loop().create_task(self._run())
        yield ClientMessage(ClientEvent.connected)

        try:
            while self._is_connected:
                yield await self._messages.get()
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect()

    async def interrupt(self):
        # NAI has no server-side cancellation.
        # We mark the current job as cancelled so that we don't report its result.
        if self._current_job is not None:
            self._current_job.state = NaiJobState.cancelled
            log.info(
                f"Marked {self._current_job} as cancelled (NAI does not support server-side cancel)"
            )

    async def cancel(self, job_ids: Iterable[str]):
        id_set = set(job_ids)
        self._queue.remove_if(lambda j: j.local_id in id_set)
        for jid in id_set:
            await self._report(ClientEvent.interrupted, jid)

    async def disconnect(self):
        if self._is_connected:
            self._is_connected = False
            if self._runner_task:
                self._runner_task.cancel()
                try:
                    await self._runner_task
                except asyncio.CancelledError:
                    pass
                self._runner_task = None

    @property
    def features(self):
        return self._features

    @property
    def performance_settings(self):
        return PerformanceSettings(
            batch_size=1,
            resolution_multiplier=settings.resolution_multiplier,
            max_pixel_count=1,  # NAI handles its own limits
            dynamic_caching=False,
            tiled_vae=False,
        )

    # -- Job execution ------------------------------------------------------

    async def _run(self):
        """Main job processing loop."""
        try:
            while self._is_connected:
                job = await self._queue.get()
                self._current_job = job
                await self._process_job(job)
                self._current_job = None
        except asyncio.CancelledError:
            pass

    async def _ensure_vibe_encodings(self, work: WorkflowInput) -> dict[int, str]:
        """Fetch (or reuse cached) vibe encodings for nai_vibe control layers.

        V4/V4.5 vibe transfer requires encodings from ai/encode-vibe; raw images
        in reference_image_multiple are not accepted. Each encode costs 2 Anlas,
        so results are cached by (image hash, model, information_extracted).
        """
        import base64
        import hashlib

        from .api import WorkflowKind
        from .resources import ControlMode

        result: dict[int, str] = {}
        cond = work.conditioning
        if cond is None or not cond.control:
            return result
        if work.kind in (WorkflowKind.inpaint, WorkflowKind.refine_region) and (
            _find_nai_base_control(work) is None
        ):
            return result  # infill drops vibes (server 500) — don't waste Anlas
        model, _cp = resolve_nai_model(work)
        if not model.supports_vibe:
            return result  # V5 launch has no Vibe Transfer; don't spend encoding Anlas
        for i, ctrl in enumerate(cond.control):
            if ctrl.mode is not ControlMode.nai_vibe or ctrl.image is None:
                continue
            image_b64 = image_to_base64(ctrl.image)
            info = round(ctrl.param2, 3)
            digest = hashlib.sha1(image_b64.encode("ascii")).hexdigest()
            key = f"{digest}|{model.value}|{info}"
            encoding = self._vibe_cache.get(key)
            if encoding is None:
                log.info(
                    f"NAI encode-vibe: encoding reference {i} (model={model.value}, ie={info})"
                )
                data = await self._post_binary(
                    "ai/encode-vibe",
                    {"image": image_b64, "model": model.value, "information_extracted": info},
                    timeout=60,
                )
                assert data is not None, "NAI encode-vibe returned an empty response"
                encoding = base64.b64encode(bytes(data)).decode("ascii")
                self._vibe_cache[key] = encoding
                _save_vibe_cache(self._vibe_cache)
            else:
                log.info(f"NAI encode-vibe: cache hit for reference {i}")
            result[i] = encoding
        return result

    async def _process_job(self, job: NaiJobInfo):
        try:
            job.state = NaiJobState.generating

            # Pre-encode vibe references (needs await, so done outside convert_workflow)
            vibe_encodings = await self._ensure_vibe_encodings(job.work)

            # Build the NAI request from WorkflowInput
            converted = convert_workflow(job.work, vibe_encodings)
            nai_request = converted.request
            job.nai_request = nai_request

            # Report progress start (NAI doesn't provide intermediate progress)
            await self._report(ClientEvent.progress, job.local_id, 0.05)

            # Log the request body, truncating base64 image/mask data for readability.
            def _truncate_for_log(obj):
                if isinstance(obj, dict):
                    return {k: _truncate_for_log(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_truncate_for_log(v) for v in obj]
                if isinstance(obj, str) and len(obj) > 200:
                    return obj[:80] + f"...({len(obj)} chars)"
                return obj

            # NOTE: infill must use the regular (non-stream) endpoint. The stream
            # endpoint (ai/generate-image-stream) IGNORES the mask for action=infill
            # and behaves like whole-image img2img (pixel-proven 2026-07-29 13:09:
            # unmasked regions were repainted too). The launcher also never streams
            # infill (its bridge and UI both fall back to non-stream for inpaint).
            log.info(
                "NAI request:\n"
                + json.dumps(_truncate_for_log(nai_request), indent=2, ensure_ascii=False)
            )
            if settings.debug_dump_workflow:
                dump_nai_request(nai_request, util.log_dir, f"nai-request-{job.local_id}")

            # Send the request
            response_data = await self._post_binary("ai/generate-image", nai_request)

            # Check if job was cancelled while waiting
            if job.state is NaiJobState.cancelled:
                await self._report(ClientEvent.interrupted, job.local_id)
                return

            # Parse the ZIP response
            images = _parse_zip_response(response_data)
            if len(images) == 0:
                raise RuntimeError("NAI returned an empty response (no images in ZIP)")

            if converted.mask_artifacts is not None:
                # Infill: the launcher never uses the raw server image. It builds
                # a transparent patch from the generated pixels and the soft
                # composite mask (add_original_image is false), so unmasked
                # pixels never change and the seam is feathered client-side.
                images = compose_infill_results(
                    job.work, images, converted.mask_artifacts, converted.focus_crop
                )
            else:
                images = restore_nai_results(job.work, images)

            job.state = NaiJobState.completed
            log.info(f"{job} completed, got {len(images)} images")
            await self._report(ClientEvent.finished, job.local_id, 1.0, images=images)

        except NetworkError as e:
            job.state = NaiJobState.failed
            log.error(
                f"{job} NetworkError: status={e.status}, code={e.code}, raw_message={e.message}"
            )
            error_msg = self._handle_nai_error(e)
            log.error(f"{job} user-facing error: {error_msg}")
            await self._report(ClientEvent.error, job.local_id, error=error_msg)

        except Exception as e:
            job.state = NaiJobState.failed
            log.exception(f"Unhandled exception while processing {job}")
            await self._report(ClientEvent.error, job.local_id, error=str(e))

    def _handle_nai_error(self, e: NetworkError) -> str:
        """Convert NAI HTTP errors into user-friendly messages."""
        status = e.status
        detail = e.message or ""
        # Try to extract JSON error detail from response body
        if detail:
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = parsed.get("message", parsed.get("error", detail))
            except (json.JSONDecodeError, TypeError):
                pass
        log.warning(f"NAI API error: status={status}, detail={detail}")

        if status == 400:
            return _("Invalid request to NovelAI API: ") + detail
        elif status == 401:
            return _(
                "NovelAI API token is invalid or expired. Please update your token in settings."
            )
        elif status == 402:
            return _("Insufficient Anlas (NovelAI credits). Please purchase more on novelai.net.")
        elif status == 429:
            return _("NovelAI rate limit exceeded. Please wait a moment and try again.")
        elif status and status >= 500:
            return (
                _("NovelAI server error") + f" ({status}): {detail}"
                if detail
                else _("NovelAI server error. Please try again later. ") + f"({status})"
            )
        else:
            return _("NovelAI API error: ") + detail

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, value, **kwargs))


# ---------------------------------------------------------------------------
# ZIP response parsing
# ---------------------------------------------------------------------------


def _parse_zip_response(data: bytes | Any) -> ImageCollection:
    """Parse a ZIP archive returned by NAI and extract images."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes from NAI API, got {type(data)}")

    images = ImageCollection()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in sorted(zf.namelist()):
                # NAI zip contains PNG files like "image_0.png"
                if name.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                    img_data = zf.read(name)
                    image = Image.from_bytes(img_data)
                    images.append(image)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"NAI returned invalid ZIP data: {e}") from e

    return images


# ---------------------------------------------------------------------------
# Request dump (parameter comparison against novelai.net)
# ---------------------------------------------------------------------------

_DUMP_MIN_PAYLOAD_CHARS = 256


def dump_nai_request(request: dict[str, Any], directory: Path, name: str) -> None:
    """Write the untruncated request to disk, images alongside it as PNGs.

    Reproducing the website exactly means comparing every field we send against
    the `Comment` chunk of an image the website produced. Neither end of that
    comparison was available before: client.log clips every string to 80
    characters, and Krita re-encodes what it saves, so plugin output carries no
    NAI metadata at all.

    Base64 payloads become a `<png len=... sha1=...>` fingerprint in the JSON and
    are decoded to `<name>.<path>.png` next to it, which is what makes "is the
    source image we sent transparent, or was it filled white?" answerable by eye.
    """
    import base64
    import hashlib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[str, str]] = []

    def strip(obj, path: str):
        if isinstance(obj, dict):
            return {k: strip(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [strip(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, str) and len(obj) > _DUMP_MIN_PAYLOAD_CHARS:
            digest = hashlib.sha1(obj.encode("ascii", "replace")).hexdigest()[:12]
            try:
                raw = base64.b64decode(obj, validate=True)
            except Exception:
                return f"<str len={len(obj)} sha1={digest}>"
            saved.append((path, digest))
            (directory / f"{name}.{path}.png").write_bytes(raw)
            return f"<png len={len(obj)} sha1={digest} file={name}.{path}.png>"
        return obj

    try:
        stripped = strip(request, "")
        target = directory / f"{name}.json"
        target.write_text(json.dumps(stripped, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"NAI request dumped to {target} ({len(saved)} image(s))")
    except Exception as e:  # never let debug tooling break a generation
        log.warning(f"Failed to dump NAI request: {e}")


# ---------------------------------------------------------------------------
# WorkflowInput → NAI request conversion
# ---------------------------------------------------------------------------


@dataclass
class NaiWorkflowRequest:
    """What convert_workflow hands to _process_job: the wire request plus the
    client-side artifacts needed to composite the result."""

    request: dict[str, Any]
    mask_artifacts: InpaintMaskArtifacts | None = None
    focus_crop: Bounds | None = None


def _find_nai_base_control(work: WorkflowInput):
    from .resources import ControlMode

    if cond := work.conditioning:
        return next(
            (
                ctrl
                for ctrl in cond.control
                if ctrl.mode is ControlMode.nai_base and ctrl.image is not None
            ),
            None,
        )
    return None


def restore_nai_results(work: WorkflowInput, images: ImageCollection) -> ImageCollection:
    """Stretch provider-sized edit results back to their exact source extent."""
    base_ctrl = _find_nai_base_control(work)
    if base_ctrl is not None and base_ctrl.image is not None:
        source_extent = base_ctrl.image.extent
    elif work.images is not None and work.images.initial_image is not None:
        source_extent = work.images.initial_image.extent
    else:
        return images
    return ImageCollection(
        image if image.extent == source_extent else Image.scale(image, source_extent)
        for image in images
    )


def compose_infill_results(
    work: WorkflowInput,
    images: ImageCollection,
    artifacts: InpaintMaskArtifacts,
    crop: Bounds | None,
) -> ImageCollection:
    """Launcher-semantics infill compositing (composeGeneratedImageArtifact).

    The generated image is resized to the crop (focused) or the canvas first and
    only then combined with the equally-resized soft composite mask, so the
    patch's RGB stays clean at the edges. The output is a canvas-sized
    transparent patch; Krita composites it over the original, which is what the
    launcher's Krita bridge sends too (transparentPatchBytes).
    """
    source = ensure(ensure(work.images).initial_image)
    canvas_extent = source.extent
    result = ImageCollection()
    for image in images:
        target = crop.extent if crop is not None else canvas_extent
        generated = Image.scale(image, target)
        alpha, _, _ = artifacts.composite_alpha_scaled(target)
        patch = apply_composite_mask(generated, alpha, target)
        if crop is not None:
            full = Image.create(canvas_extent, fill=0)  # transparent
            full.draw_image(patch, crop.offset)
            result.append(full)
        else:
            result.append(patch)
    return result


def restore_nai_base_results(work: WorkflowInput, images: ImageCollection) -> ImageCollection:
    return restore_nai_results(work, images)


def _find_style_for_checkpoint(checkpoint: str):
    """Find the Style whose checkpoints list contains the given checkpoint ID."""
    from ..style import Styles

    for s in Styles.list():
        if checkpoint in s.checkpoints:
            return s
    return None


def resolve_nai_model(work: WorkflowInput) -> tuple[NaiModel, str]:
    """Resolve the NAI model for a workflow (checkpoint id or settings fallback)."""
    model = NaiModel.default()
    checkpoint_id = ""
    if work.models and work.models.checkpoint:
        checkpoint_id = work.models.checkpoint
        for nai_m in NaiModel:
            if nai_m.value == checkpoint_id:
                model = nai_m
                break
    if checkpoint_id == "":
        try:
            model = NaiModel(settings.nai_model)
        except ValueError:
            pass
    return model, checkpoint_id


def convert_workflow(
    work: WorkflowInput, vibe_encodings: dict[int, str] | None = None
) -> NaiWorkflowRequest:
    """Convert a ``WorkflowInput`` into a NAI API request body.

    Reads NAI-specific generation parameters from the matching Style preset.
    Falls back to global settings defaults when no style is found.
    Image/mask/extent/seed/strength come from the WorkflowInput.

    ``vibe_encodings`` maps indices into ``work.conditioning.control`` to
    pre-fetched vibe encodings (from ai/encode-vibe, see NaiClient).
    """
    from ..text import merge_prompt
    from .api import WorkflowKind

    # --- Prompt ---
    cond = work.conditioning
    prompt = cond.positive if cond else ""
    negative = cond.negative if cond else ""
    style_prompt = cond.style if cond else ""

    # Use merge_prompt to correctly substitute {prompt} placeholder in style template
    prompt = merge_prompt(prompt, style_prompt)

    base_ctrl = _find_nai_base_control(work)

    # --- Action ---
    if base_ctrl is not None:
        action = NaiAction.img2img
    elif work.kind is WorkflowKind.inpaint or work.kind is WorkflowKind.refine_region:
        action = NaiAction.infill
    elif (
        work.kind is WorkflowKind.refine
        or work.images
        and work.images.initial_image is not None
        and work.kind is not WorkflowKind.generate
    ):
        action = NaiAction.img2img
    else:
        action = NaiAction.generate

    # --- Resolution (from WorkflowInput) ---
    # The docker's target resolution is global: text to image, img2img and inpaint
    # all render at it. Only when the workflow carries no target at all do the
    # fallbacks below apply.
    if base_ctrl is not None:
        assert base_ctrl.image is not None
        extent = base_ctrl.target_extent or base_ctrl.image.extent
        log.info(f"NAI img2img resolution: {extent.width}x{extent.height}")
    elif work.nai_target_extent is not None:
        # Text generation used to skip this branch and fall through to
        # extent.desired below - the ComfyUI-style extent, which is capped near
        # 1 MP - so a 1472x1472 canvas came back as 1024x1024 however the docker
        # was set.
        extent = work.nai_target_extent
        log.info(f"NAI target resolution: {extent.width}x{extent.height}")
    elif action in (NaiAction.img2img, NaiAction.infill):
        if work.images and work.images.initial_image:
            extent = nai_auto_resolution(work.images.initial_image.extent)
        else:
            extent = Extent(1024, 1024)
        log.info(f"NAI redraw resolution: {extent.width}x{extent.height}")
    elif work.images and work.images.initial_image:
        extent = work.images.initial_image.extent
        log.info(f"NAI resolution: using initial_image extent {extent.width}x{extent.height}")
    elif work.images:
        extent = work.images.extent.desired
        log.info(f"NAI resolution: using extent.desired {extent.width}x{extent.height}")
    else:
        extent = Extent(1024, 1024)
    max_pixels = (
        NAI_EDIT_MAX_PIXELS if action in (NaiAction.img2img, NaiAction.infill) else NAI_MAX_PIXELS
    )
    extent = clamp_resolution(extent, max_pixels)

    # --- Model (resolve before action so we know v3 vs v4) ---
    model, checkpoint_id = resolve_nai_model(work)

    # --- NAI control layers (img2img base / vibe transfer / precise reference) ---
    from .resources import ControlMode

    vibe_ctrls: list[tuple[int, Any]] = []  # (index into cond.control, ControlInput)
    precise_ctrls: list[Any] = []
    if cond:
        for i, ctrl in enumerate(cond.control):
            if ctrl.image is None:
                continue
            if ctrl.mode is ControlMode.nai_vibe:
                vibe_ctrls.append((i, ctrl))
            elif ctrl.mode.is_nai_precise:
                precise_ctrls.append(ctrl)

    # Server-side compatibility rules (mirrors the launcher):
    if precise_ctrls and not model.supports_precise_reference:
        log.warning("NAI Precise Reference requires a V4.5 model, dropping references")
        precise_ctrls = []
    if vibe_ctrls and not model.supports_vibe:
        log.warning("NAI V5 does not support Vibe Transfer yet, dropping vibes")
        vibe_ctrls = []
    if precise_ctrls and vibe_ctrls:
        log.warning("NAI Precise Reference and Vibe Transfer are incompatible, dropping vibes")
        vibe_ctrls = []
    if action is NaiAction.infill and vibe_ctrls:
        log.warning("NAI infill does not support Vibe Transfer (server error 500), dropping vibes")
        vibe_ctrls = []

    # --- Look up the Style preset ---
    # Prefer the exact style threaded from the UI (work.nai_style is the unique
    # filename of the selected style). Fall back to guessing by checkpoint only
    # for legacy inputs that predate the threaded field.
    from ..style import Styles

    style = None
    if getattr(work, "nai_style", ""):
        style = Styles.list().find(work.nai_style)
    if style is None and checkpoint_id:
        style = _find_style_for_checkpoint(checkpoint_id)

    # --- Sampling (from style preset → sampler preset → fallback settings) ---
    # Steps & CFG come from the style's sampler preset
    from ..style import SamplerPresets

    sampler_preset_name = style.sampler if style else None
    sampler_preset = None
    if sampler_preset_name:
        try:
            sampler_preset = SamplerPresets.instance()[sampler_preset_name]
        except KeyError:
            pass

    if sampler_preset:
        # Map ComfyUI sampler name → NAI sampler
        sampler = (
            map_sampler(sampler_preset.sampler)
            if hasattr(sampler_preset, "sampler")
            else NaiSampler.default()
        )
        noise_schedule = (
            map_noise_schedule(sampler_preset.scheduler)
            if hasattr(sampler_preset, "scheduler")
            else NaiNoiseSchedule.default()
        )
        steps = style.sampler_steps if style else settings.nai_steps
        cfg_scale = style.cfg_scale if style else settings.nai_cfg_scale
    else:
        # Fallback: try NAI sampler from global settings
        try:
            sampler = NaiSampler(settings.nai_sampler)
        except ValueError:
            sampler = NaiSampler.default()
        try:
            noise_schedule = NaiNoiseSchedule(settings.nai_noise_schedule)
        except ValueError:
            noise_schedule = NaiNoiseSchedule.default()
        steps = settings.nai_steps
        cfg_scale = settings.nai_cfg_scale

    # --- NAI-specific params (from style, fallback to settings) ---
    if style and any(cp.startswith("nai-diffusion") for cp in style.checkpoints):
        cfg_rescale_val = style.nai_cfg_rescale
        try:
            uc_preset = NaiUCPreset(style.nai_uc_preset)
        except ValueError:
            uc_preset = NaiUCPreset.heavy
        quality_toggle = style.nai_quality_toggle
        quality_tier = style.nai_quality_tier
        variety_plus: bool = bool(style.nai_variety_boost)
        # Override noise_schedule from style if explicitly set
        try:
            ns = NaiNoiseSchedule(style.nai_noise_schedule)
            noise_schedule = ns
        except ValueError:
            pass
    else:
        cfg_rescale_val = settings.nai_cfg_rescale
        try:
            uc_preset = NaiUCPreset(settings.nai_uc_preset)
        except ValueError:
            uc_preset = NaiUCPreset.heavy
        quality_toggle = settings.nai_quality_toggle
        # No global counterpart: the tier is a style-level choice, and the
        # builder falls back to standard for any model without light anyway.
        quality_tier = QualityTags.standard_tier
        variety_plus = bool(settings.nai_variety_boost)

    # Seed comes from WorkflowInput (per-generation from UI)
    seed = work.sampling.seed if work.sampling else 0

    # The plugin's NaiUCPreset enum predates the registry port and has its own
    # numbering (none=2, which the SERVER reads as humanFocus) — map it to the
    # official API values before it goes anywhere near a request.
    uc_preset_api = {
        NaiUCPreset.heavy: UcPresets.heavy_api_value,
        NaiUCPreset.light: UcPresets.light_api_value,
        NaiUCPreset.none: UcPresets.none_api_value,
    }[uc_preset]

    # --- img2img / inpaint sources (from WorkflowInput) ---
    src: Image | None = None
    mask_img: Image | None = None
    strength = 0.7
    noise_val = 0.0
    inpaint_strength = 1.0
    # Focused inpaint: send only the focus box plus its context margin. The
    # masked area then gets the whole request resolution instead of the few
    # pixels it would occupy in a downscaled full canvas; compose_infill_results
    # puts the patch back at the crop offset.
    focus_crop = work.nai_focus_crop if base_ctrl is None else None

    if action in (NaiAction.img2img, NaiAction.infill):
        if base_ctrl is not None:
            src = base_ctrl.image
        elif work.images and work.images.initial_image:
            src = work.images.initial_image
        # The source keeps its alpha. NAI accepts RGBA and the launcher never
        # strips it (no makeOpaque/removeAlpha anywhere in its pipeline);
        # filling transparency with white handed the model a different picture
        # than the canvas held, which is what made img2img drift ~1.7x further
        # from the source than the website did and come back with white corners
        # where the website returned transparent ones. The defensive copy went
        # with it - nothing mutates src in place any more.
        if src is not None and focus_crop is not None:
            src = Image.crop(src, focus_crop)
        if action is NaiAction.infill and work.images and work.images.hires_mask:
            mask_img = work.images.hires_mask
            if focus_crop is not None:
                mask_img = Image.crop(mask_img, focus_crop)

    if action is NaiAction.img2img:
        if base_ctrl is not None:
            # img2img via a base-image control layer: strength/noise come from the
            # layer's OWN knobs (launcher defaults 0.7 / 0.0). The main strength
            # slider only applies to redraw workflows, matching NAI web's separate
            # img2img UI.
            strength = base_ctrl.strength
            noise_val = base_ctrl.param2
        else:
            # Whole-canvas refine: main slider -> `strength`.
            if work.sampling:
                strength = work.sampling.denoise_strength
            noise_val = 0.0
        if strength >= 1.0:  # avoid degenerate full replacement at the slider max
            strength = 0.99
        strength = max(strength, 0.01)

    if action is NaiAction.infill:
        # 重绘幅度 -> inpaintImg2ImgStrength (and, below 100%, the nested img2img
        # object with color_correct). The flat `strength` is sent too, exactly as
        # the launcher and the web UI do; the server ignores it for infill.
        if work.sampling:
            inpaint_strength = work.sampling.denoise_strength
            strength = inpaint_strength
        noise_val = 0.0

    # --- Vibe Transfer ---
    # V4/V4.5 requires PRE-ENCODED vibes (ai/encode-vibe) in reference_image_multiple,
    # never raw images; V3 sends the raw image directly. Encodings are fetched (and
    # cached) in NaiClient._process_job and passed in via vibe_encodings by index.
    vibes: list[VibeReference] = []
    for i, ctrl in vibe_ctrls:
        if model.is_v3:
            assert ctrl.image is not None
            vibes.append(VibeReference(image_to_base64(ctrl.image), ctrl.strength, ctrl.param2))
            continue
        encoding = (vibe_encodings or {}).get(i)
        if encoding is None:
            log.warning("NAI: missing vibe encoding for control layer %d, skipping", i)
            continue
        vibes.append(VibeReference(encoding, ctrl.strength, ctrl.param2))

    # --- Precise (director) reference — V4.5 only ---
    precise_refs = [
        PreciseReference(
            image_b64=image_to_base64(prepare_nai_precise_reference_image(ctrl.image)),
            type_caption=ctrl.mode.nai_precise_caption,
            strength=ctrl.strength,
            # Launcher: secondary strength = 1.0 - fidelity; the builder inverts.
            fidelity=ctrl.param2,
        )
        for ctrl in precise_ctrls
    ]

    # --- Build the launcher-contract request ---
    params = NaiGenerationParams(
        model=model.value,
        action={
            NaiAction.generate: ACTION_GENERATE,
            NaiAction.img2img: ACTION_IMG2IMG,
            NaiAction.infill: ACTION_INFILL,
        }[action],
        width=extent.width,
        height=extent.height,
        prompt=prompt,
        negative_prompt=negative,
        scale=cfg_scale,
        sampler=sampler.value,
        steps=steps,
        # Always 1, never work.batch_count. That field is ComfyUI's local
        # micro-batching: compute_batch_size packs 2-4 latents into one run when
        # the extent is small, which is free on your own GPU and meaningless
        # here - NAI bills per sample. The batch slider already asks for more
        # images by enqueuing more jobs, so honouring it here multiplied on top
        # (512px canvas + slider 4 = 16 images charged for one click), and the
        # extra samples are noised server-side, so none is reproducible at the
        # requested seed. The launcher forces this in all 10 call sites,
        # krita_bridge_service.dart:253 and anlas_calculator.dart:139 included.
        n_samples=1,
        seed=seed,
        uc_preset=uc_preset_api,
        quality_toggle=quality_toggle,
        quality_tier=quality_tier,
        cfg_rescale=cfg_rescale_val,
        noise_schedule=noise_schedule.value,
        variety_plus=variety_plus,
        # Account-level on the website and independent of the transparency tag;
        # sending false made the server return premultiplied alpha, which darkened
        # every semi-transparent edge pixel by its own alpha.
        straight_alpha=True,
        source_image=src,
        mask_image=mask_img,
        strength=strength,
        noise=noise_val,
        inpaint_strength=inpaint_strength,
        vibes=vibes,
        precise_references=precise_refs,
    )
    result = build_request(params)
    return NaiWorkflowRequest(result.request_data, result.mask_artifacts, focus_crop)


# ---------------------------------------------------------------------------
# NAI model list (static — NAI models are fixed, not user-configurable)
# ---------------------------------------------------------------------------


def _build_nai_models() -> ClientModels:
    """Build a ``ClientModels`` instance representing NAI's fixed model set.

    Since NAI doesn't expose individual checkpoint / LoRA / VAE files,
    we populate the ``checkpoints`` dict with NAI model identifiers so
    that the existing style / UI code can reference them.
    """
    from ..files import FileFormat
    from .client import CheckpointInfo
    from .resources import Arch

    models = ClientModels()
    # Register NAI models as "checkpoints" so style resolution works
    for nai_model in NaiModel.list_generate():
        models.checkpoints[nai_model.value] = CheckpointInfo(
            filename=nai_model.value,
            arch=Arch.nai,
            format=FileFormat.checkpoint,
        )
    return models
