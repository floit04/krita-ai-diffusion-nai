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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PyQt5.QtNetwork import QNetworkReply

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
from ..image import Image, ImageCollection
from ..localization import translate as _
from .network import NetworkError, RequestManager
from .nai_workflow import (
    NaiAction,
    NaiModel,
    NaiNoiseSchedule,
    NaiSampler,
    NaiUCPreset,
    apply_quality_tags,
    build_generate_request,
    clamp_resolution,
    composite_nai_patch,
    image_to_base64,
    map_sampler,
    map_noise_schedule,
    prepare_nai_precise_reference_image,
    prepare_nai_request_mask,
)
from ..settings import PerformanceSettings, settings
from ..util import client_logger as log


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
        return self

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
        return await self._requests.post(
            f"{self.url}/{path}", data, bearer=self._current_token()
        )

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
            log.info(f"Marked {self._current_job} as cancelled (NAI does not support server-side cancel)")

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
        if work.kind in (WorkflowKind.inpaint, WorkflowKind.refine_region):
            return result  # infill drops vibes (server 500) — don't waste Anlas
        model, _cp = resolve_nai_model(work)
        for i, ctrl in enumerate(cond.control):
            if ctrl.mode is not ControlMode.nai_vibe or ctrl.image is None:
                continue
            image_b64 = image_to_base64(ctrl.image)
            info = round(ctrl.param2, 3)
            digest = hashlib.sha1(image_b64.encode("ascii")).hexdigest()
            key = f"{digest}|{model.value}|{info}"
            encoding = self._vibe_cache.get(key)
            if encoding is None:
                log.info(f"NAI encode-vibe: encoding reference {i} (model={model.value}, ie={info})")
                data = await self._post_binary(
                    "ai/encode-vibe",
                    {"image": image_b64, "model": model.value, "information_extracted": info},
                    timeout=60,
                )
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
            nai_request = convert_workflow(job.work, vibe_encodings)
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

            # Inpaint/refine_region: NAI regenerates the whole canvas, so composite
            # each result into a transparent patch whose alpha is the selection mask.
            # Written back, the original shows through outside the selection (no
            # colour drift, no black border); only the masked area changes.
            if job.work.images and job.work.images.hires_mask is not None:
                mask_img = job.work.images.hires_mask
                images = ImageCollection(
                    composite_nai_patch(im, mask_img, mask_img.extent) for im in images
                )

            job.state = NaiJobState.completed
            log.info(f"{job} completed, got {len(images)} images")
            await self._report(
                ClientEvent.finished, job.local_id, 1.0, images=images
            )

        except NetworkError as e:
            job.state = NaiJobState.failed
            log.error(f"{job} NetworkError: status={e.status}, code={e.code}, raw_message={e.message}")
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
            return _("NovelAI API token is invalid or expired. Please update your token in settings.")
        elif status == 402:
            return _("Insufficient Anlas (NovelAI credits). Please purchase more on novelai.net.")
        elif status == 429:
            return _("NovelAI rate limit exceeded. Please wait a moment and try again.")
        elif status and status >= 500:
            return _("NovelAI server error") + f" ({status}): {detail}" if detail else _("NovelAI server error. Please try again later. ") + f"({status})"
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
# WorkflowInput → NAI request conversion
# ---------------------------------------------------------------------------


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
) -> dict[str, Any]:
    """Convert a ``WorkflowInput`` into a NAI API request body.

    Reads NAI-specific generation parameters from the matching Style preset.
    Falls back to global settings defaults when no style is found.
    Image/mask/extent/seed/strength come from the WorkflowInput.

    ``vibe_encodings`` maps indices into ``work.conditioning.control`` to
    pre-fetched vibe encodings (from ai/encode-vibe, see NaiClient).
    """
    from .api import WorkflowKind
    from ..text import merge_prompt

    # --- Prompt ---
    cond = work.conditioning
    prompt = cond.positive if cond else ""
    negative = cond.negative if cond else ""
    style_prompt = cond.style if cond else ""

    # Use merge_prompt to correctly substitute {prompt} placeholder in style template
    prompt = merge_prompt(prompt, style_prompt)

    # --- Resolution (from WorkflowInput) ---
    # For img2img/inpaint, the image has already been cropped to bbox by model.py.
    # We must use the actual image dimensions (not extent.desired which may differ
    # due to ComfyUI 2-pass resolution scaling that NAI doesn't use).
    if work.images and work.images.initial_image:
        extent = work.images.initial_image.extent
        log.info(f"NAI resolution: using initial_image extent {extent.width}x{extent.height}")
    elif work.images:
        extent = work.images.extent.desired
        log.info(f"NAI resolution: using extent.desired {extent.width}x{extent.height}")
    else:
        from ..image import Extent
        extent = Extent(1024, 1024)
    extent = clamp_resolution(extent)

    # --- Model (resolve before action so we know v3 vs v4) ---
    model, checkpoint_id = resolve_nai_model(work)

    # --- NAI control layers (img2img base / vibe transfer / precise reference) ---
    from .resources import ControlMode

    base_ctrl = None
    vibe_ctrls: list[tuple[int, Any]] = []  # (index into cond.control, ControlInput)
    precise_ctrls: list[Any] = []
    if cond:
        for i, ctrl in enumerate(cond.control):
            if ctrl.image is None:
                continue
            if ctrl.mode is ControlMode.nai_base and base_ctrl is None:
                base_ctrl = ctrl
            elif ctrl.mode is ControlMode.nai_vibe:
                vibe_ctrls.append((i, ctrl))
            elif ctrl.mode.is_nai_precise:
                precise_ctrls.append(ctrl)

    # --- Action ---
    if work.kind is WorkflowKind.inpaint:
        # All NAI models use infill for inpainting (v3 switches to inpaint model variant)
        action = NaiAction.infill
    elif work.kind is WorkflowKind.refine_region:
        # refine_region with mask: also use infill for proper mask-based inpainting
        action = NaiAction.infill
    elif work.kind in (WorkflowKind.refine,):
        action = NaiAction.img2img
    elif work.images and work.images.initial_image is not None and work.kind is not WorkflowKind.generate:
        action = NaiAction.img2img
    else:
        action = NaiAction.generate

    # An img2img base layer (垫图) turns plain generation into img2img with that
    # image as source. A selection (mask) takes priority: infill stays untouched.
    if action is NaiAction.generate and base_ctrl is not None:
        action = NaiAction.img2img
    if action is NaiAction.infill and base_ctrl is not None:
        log.info("NAI: selection redraw active, ignoring img2img base layer")
        base_ctrl = None

    # Server-side compatibility rules (mirrors the launcher):
    if precise_ctrls and not model.is_v4_5:
        log.warning("NAI Precise Reference requires a V4.5 model, dropping references")
        precise_ctrls = []
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
        sampler = map_sampler(sampler_preset.sampler) if hasattr(sampler_preset, 'sampler') else NaiSampler.default()
        noise_schedule = map_noise_schedule(sampler_preset.scheduler) if hasattr(sampler_preset, 'scheduler') else NaiNoiseSchedule.default()
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
        variety_plus = bool(settings.nai_variety_boost)

    # Seed comes from WorkflowInput (per-generation from UI)
    seed = work.sampling.seed if work.sampling else 0

    # --- img2img / inpaint (from WorkflowInput) ---
    image_b64 = None
    mask_b64 = None
    strength = 0.7
    noise_val = 0.0
    # 重绘幅度 drives `strength` for both img2img and infill (see below).
    # inpaintImg2ImgStrength is inert in our NAI requests, so it stays None and is
    # omitted from the request entirely.
    inpaint_strength: float | None = None
    # Both inpaint and refine_region carry a mask.
    needs_mask = work.kind in (WorkflowKind.inpaint, WorkflowKind.refine_region)

    if action in (NaiAction.img2img, NaiAction.infill):
        # Source image: NAI's img2img/infill requires the image to match the
        # requested width/height, so scale (stretch) it to the request extent first
        # — same as the launcher's normalizeImageForRequest. An img2img base layer
        # replaces the canvas as the source.
        src = None
        if base_ctrl is not None:
            src = base_ctrl.image
        elif work.images and work.images.initial_image:
            src = work.images.initial_image
        if src is not None:
            if src.extent != extent:
                src = Image.scale(src, extent)
            src.make_opaque()  # NAI requires RGB, strip alpha
            image_b64 = image_to_base64(src)
        # Mask: a clean binary mask aligned to NAI's 8px latent grid. No
        # feather/grow/blend — Krita's ComfyUI-oriented preprocessing corrupts NAI
        # inpaint (redraws a wrong-shaped region with black borders). See
        # prepare_nai_request_mask; the result is composited client-side in
        # _process_job so unmasked pixels keep the exact original.
        if needs_mask and work.images and work.images.hires_mask:
            req_mask = prepare_nai_request_mask(work.images.hires_mask, extent)
            mask_b64 = image_to_base64(req_mask)

    if action is NaiAction.img2img:
        if base_ctrl is not None:
            # img2img via a base-image control layer: strength/noise come from the
            # layer's OWN knobs (launcher defaults 0.7 / 0.0). The main strength
            # slider only applies to redraw workflows, matching NAI web's separate
            # img2img UI.
            strength = base_ctrl.strength
            noise_val = base_ctrl.param2
        else:
            # Whole-canvas refine: main slider -> `strength`. Verified working
            # end-to-end via client.log + user tests on the non-stream endpoint.
            if work.sampling:
                strength = work.sampling.denoise_strength
            noise_val = 0.0
        if strength >= 1.0:  # avoid degenerate full replacement at the slider max
            strength = 0.99
        if strength < 0.01:
            strength = 0.01

    if action is NaiAction.infill:
        # Masked redraw follows the working reference implementation
        # (ComfyUI_RS_NAI_API_Request NAIInpaintNode) exactly:
        #   inpaintImg2ImgStrength = 重绘幅度 slider
        #   add_original_image     = true  (set in build_generate_request)
        #   noise                  = 0.0
        #   `strength`             = NOT SENT (build_generate_request omits it for
        #                            infill; it belongs to action=img2img only)
        # The launcher's wiring (strength=0.7 + add_original_image=false) is NOT
        # honored by the server (pixel-proven full repaint at any value).
        if work.sampling:
            inpaint_strength = work.sampling.denoise_strength
        noise_val = 0.0

    # --- Vibe Transfer ---
    # V4/V4.5 requires PRE-ENCODED vibes (ai/encode-vibe) in reference_image_multiple,
    # never raw images. Encodings are fetched (and cached) in NaiClient._process_job
    # and passed in via vibe_encodings, keyed by control index.
    ref_images: list[str] = []
    ref_strengths: list[float] = []
    ref_info_extracted: list[float] = []
    for i, ctrl in vibe_ctrls:
        encoding = (vibe_encodings or {}).get(i)
        if encoding is None:
            log.warning("NAI: missing vibe encoding for control layer %d, skipping", i)
            continue
        ref_images.append(encoding)
        ref_strengths.append(ctrl.strength)
        ref_info_extracted.append(ctrl.param2)

    # --- Precise (director) reference — V4.5 only ---
    precise_refs: list[dict[str, Any]] = []
    for ctrl in precise_ctrls:
        ref_img = prepare_nai_precise_reference_image(ctrl.image)
        precise_refs.append({
            "image": image_to_base64(ref_img),
            "caption": ctrl.mode.nai_precise_caption,
            "strength": ctrl.strength,
            # Launcher: secondary strength = 1.0 - fidelity (note the inversion).
            "secondary": 1.0 - ctrl.param2,
        })

    # --- Quality tags ---
    # When "Add Quality Tags" is on, append the model-specific quality tags to the
    # (already style-merged, post-positioned) prompt. NAI reads the toggle state
    # from the tags' presence in the prompt, so appending them is what actually
    # turns "Add Quality Tags" on for the generated image. Must run BEFORE the
    # prompt is consumed by v4_prompt_obj and build_generate_request below.
    if quality_toggle:
        prompt = apply_quality_tags(prompt, model)
        log.info(f"NAI quality tags appended for {model.value}")

    # --- V4/V4.5 structured prompt ---
    v4_prompt_obj = None
    v4_negative_obj = None
    if model.is_v4:
        v4_prompt_obj = {
            "caption": {
                "base_caption": prompt,
                "char_captions": [],
            },
            "use_coords": False,
            "use_order": True,
        }
        v4_negative_obj = {
            "caption": {
                "base_caption": negative,
                "char_captions": [],
            },
            "legacy_uc": False,
        }

    # --- Build request ---
    request = build_generate_request(
        prompt=prompt,
        negative_prompt=negative,
        width=extent.width,
        height=extent.height,
        model=model,
        action=action,
        sampler=sampler,
        steps=steps,
        scale=cfg_scale,
        cfg_rescale=cfg_rescale_val,
        noise_schedule=noise_schedule,
        seed=seed,
        n_samples=work.batch_count,
        quality_toggle=quality_toggle,
        uc_preset=uc_preset,
        variety_plus=variety_plus,
        image=image_b64,
        strength=strength,
        noise=noise_val,
        mask=mask_b64,
        inpaint_img2img_strength=inpaint_strength,
        reference_image_multiple=ref_images if ref_images else None,
        reference_strength_multiple=ref_strengths if ref_strengths else None,
        reference_information_extracted_multiple=ref_info_extracted if ref_info_extracted else None,
        precise_references=precise_refs if precise_refs else None,
        v4_prompt=v4_prompt_obj,
        v4_negative_prompt=v4_negative_obj,
    )

    return request


# ---------------------------------------------------------------------------
# NAI model list (static — NAI models are fixed, not user-configurable)
# ---------------------------------------------------------------------------


def _build_nai_models() -> ClientModels:
    """Build a ``ClientModels`` instance representing NAI's fixed model set.

    Since NAI doesn't expose individual checkpoint / LoRA / VAE files,
    we populate the ``checkpoints`` dict with NAI model identifiers so
    that the existing style / UI code can reference them.
    """
    from .client import CheckpointInfo
    from ..files import FileFormat
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
