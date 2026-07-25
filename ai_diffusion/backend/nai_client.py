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
    build_generate_request,
    clamp_resolution,
    image_to_base64,
    map_sampler,
    map_noise_schedule,
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
            max_control_layers=0,
        )

    # -- HTTP helpers -------------------------------------------------------

    async def _get(self, path: str, timeout: float | None = 30):
        return await self._requests.get(
            f"{self.url}/{path}", timeout=timeout, bearer=self._token
        )

    async def _post(self, path: str, data: dict, timeout: float | None = None):
        return await self._requests.post(
            f"{self.url}/{path}", data, bearer=self._token
        )

    async def _post_binary(self, path: str, data: dict, timeout: float | None = 300):
        """POST that expects a binary (ZIP) response."""
        return await self._requests.http(
            "POST", f"{self.url}/{path}", data, timeout=timeout, bearer=self._token
        )

    async def _validate_token(self):
        """Lightweight request to verify that the API token is valid."""
        await self._get(
            "ai/generate-image/suggest-tags?model=nai-diffusion-3&prompt=test",
            timeout=15,
        )

    # -- Client ABC implementation ------------------------------------------

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

    async def _process_job(self, job: NaiJobInfo):
        try:
            job.state = NaiJobState.generating

            # Build the NAI request from WorkflowInput
            nai_request = convert_workflow(job.work)
            job.nai_request = nai_request

            # Report progress start (NAI doesn't provide intermediate progress)
            await self._report(ClientEvent.progress, job.local_id, 0.05)

            # Log the FULL request body (truncate base64 data for readability)
            def _truncate_for_log(obj):
                if isinstance(obj, dict):
                    return {k: _truncate_for_log(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_truncate_for_log(v) for v in obj]
                if isinstance(obj, str) and len(obj) > 200:
                    return obj[:80] + f"...({len(obj)} chars)"
                return obj

            log.warning(f"NAI FULL REQUEST:\n{json.dumps(_truncate_for_log(nai_request), indent=2, ensure_ascii=False)}")

            # Detailed inpaint/mask diagnostic
            params = nai_request.get("parameters", {})
            log.warning(f"=== NAI INPAINT DIAGNOSTIC ===")
            log.warning(f"action: {nai_request.get('action')}")
            log.warning(f"width: {params.get('width')}, height: {params.get('height')}")
            log.warning(f"strength: {params.get('strength')}")
            log.warning(f"add_original_image: {params.get('add_original_image')}")
            log.warning(f"work.kind: {job.work.kind}")

            # Decode and inspect image
            img_b64 = params.get("image", "")
            log.warning(f"image base64 length: {len(img_b64)}")
            if img_b64:
                try:
                    import base64 as b64mod
                    from PIL import Image as PILImage
                    import numpy as np
                    _img = PILImage.open(io.BytesIO(b64mod.b64decode(img_b64)))
                    log.warning(f"image decoded size: {_img.size}, mode: {_img.mode}")
                except Exception as _e:
                    log.warning(f"image decode failed: {_e}")

            # Decode and inspect mask
            mask_b64 = params.get("mask", "")
            log.warning(f"mask base64 length: {len(mask_b64)}")
            if mask_b64:
                try:
                    import base64 as b64mod
                    from PIL import Image as PILImage
                    import numpy as np
                    _mask = PILImage.open(io.BytesIO(b64mod.b64decode(mask_b64)))
                    log.warning(f"mask decoded size: {_mask.size}, mode: {_mask.mode}")
                    _arr = np.array(_mask)
                    log.warning(f"mask shape: {_arr.shape}")
                    log.warning(f"mask mean: {_arr.mean():.2f}")
                    log.warning(f"mask min/max: {_arr.min()}/{_arr.max()}")
                    _white = (_arr > 200).sum() / _arr.size
                    log.warning(f"mask white ratio: {_white:.4f}")
                except Exception as _e:
                    log.warning(f"mask decode failed: {_e}")
            else:
                log.warning("!!! NO MASK in request !!!")

            if job.work.images:
                log.warning(f"initial_image present: {job.work.images.initial_image is not None}")
                log.warning(f"hires_mask present: {job.work.images.hires_mask is not None}")
                if job.work.images.initial_image:
                    log.warning(f"initial_image size: {job.work.images.initial_image.extent}")
                if job.work.images.hires_mask:
                    log.warning(f"hires_mask size: {job.work.images.hires_mask.extent}")

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

            # --- TEMP DIAGNOSTIC, REMOVE LATER — save raw NAI output before any post-processing ---
            try:
                _debug_path = r"C:\Users\Tu\Desktop\nai_debug_output.png"
                if len(images) > 0:
                    images[0].save(_debug_path)
                    log.warning(f"TEMP DIAGNOSTIC: saved NAI raw output to {_debug_path} ({images[0].extent})")
            except Exception as _save_err:
                log.warning(f"TEMP DIAGNOSTIC: failed to save NAI raw output: {_save_err}")
            # --- END TEMP DIAGNOSTIC ---

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


def _nai_feather_mask(mask_img, grow: int, feather: int):
    """Apply grow (dilation) and feather (blur) to mask for smooth inpaint transitions.

    NAI requires feathered mask, like ComfyUI path (apply_grow_feather).
    Uses Qt scale operations to approximate Gaussian blur on the mask edges.
    Without this, the raw binary mask produces visible gray edges at inpaint boundaries.
    """
    total_radius = grow + feather
    if total_radius <= 0:
        return mask_img

    import math
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage
    from ..image import Image

    qimg = QImage(mask_img._qimage)  # copy to avoid modifying original
    w, h = qimg.width(), qimg.height()
    if w <= 0 or h <= 0:
        return mask_img

    # Scale-down then scale-up approximates a box blur.
    # With 3 passes the effective blur approaches a Gaussian.
    # Compute the downscale size so the blur radius ≈ total_radius pixels.
    scale_ratio = w * math.sqrt(3) / (2.0 * total_radius)
    small_w = max(4, int(round(scale_ratio)))
    small_h = max(4, int(round(h * small_w / w)))

    mode = Qt.TransformationMode.SmoothTransformation
    aspect = Qt.AspectRatioMode.IgnoreAspectRatio
    for _ in range(3):
        qimg = qimg.scaled(small_w, small_h, aspect, mode)
        qimg = qimg.scaled(w, h, aspect, mode)

    # Ensure format stays as Grayscale8
    if qimg.format() != QImage.Format.Format_Grayscale8:
        qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)

    log.info(f"NAI mask feathering applied: grow={grow}, feather={feather}, total_radius={total_radius}")
    return Image(qimg)


def convert_workflow(work: WorkflowInput) -> dict[str, Any]:
    """Convert a ``WorkflowInput`` into a NAI API request body.

    Reads NAI-specific generation parameters from the matching Style preset.
    Falls back to global settings defaults when no style is found.
    Image/mask/extent/seed/strength come from the WorkflowInput.
    """
    from .api import WorkflowKind
    from ..text import merge_prompt

    # --- Prompt ---
    cond = work.conditioning
    prompt = cond.positive if cond else ""
    negative = cond.negative if cond else ""
    style_prompt = cond.style if cond else ""

    # Debug: show all conditioning fields including regions
    if cond:
        regions_info = [(r.positive, r.negative if hasattr(r, 'negative') else '?') for r in cond.regions] if cond.regions else []
        log.warning(f"NAI PROMPT DEBUG: cond.positive={repr(prompt)}, cond.style={repr(style_prompt)}, regions={regions_info}, language={repr(cond.language)}")
    else:
        log.warning("NAI PROMPT DEBUG: cond is None!")

    # Use merge_prompt to correctly substitute {prompt} placeholder in style template
    prompt = merge_prompt(prompt, style_prompt)

    log.warning(f"NAI PROMPT AFTER MERGE: prompt={repr(prompt)}")

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

    # --- Look up the Style preset matching this checkpoint ---
    style = _find_style_for_checkpoint(checkpoint_id) if checkpoint_id else None

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
        skip_sigma: float | None = settings.nai_variety_boost_sigma if style.nai_variety_boost else None
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
        skip_sigma = settings.nai_variety_boost_sigma if settings.nai_variety_boost else None

    # Seed comes from WorkflowInput (per-generation from UI)
    seed = work.sampling.seed if work.sampling else 0

    # --- img2img / inpaint (from WorkflowInput) ---
    image_b64 = None
    mask_b64 = None
    strength = 0.7
    noise_val = 0.0
    # --- TEMP DIAGNOSTIC: STRENGTH PATH (REMOVE LATER) ---
    if work.sampling:
        log.warning(f"TEMP DIAGNOSTIC: STRENGTH PATH [5] convert_workflow: work.kind={work.kind}, action={action}, sampling.total_steps={work.sampling.total_steps}, sampling.start_step={work.sampling.start_step}, sampling.denoise_strength={work.sampling.denoise_strength}")
    else:
        log.warning(f"TEMP DIAGNOSTIC: STRENGTH PATH [5] convert_workflow: work.kind={work.kind}, action={action}, sampling=None")
    # --- END TEMP DIAGNOSTIC ---
    # Both inpaint and refine_region use mask
    needs_mask = work.kind in (WorkflowKind.inpaint, WorkflowKind.refine_region)

    if action is NaiAction.img2img:
        if work.images and work.images.initial_image:
            work.images.initial_image.make_opaque()  # NAI requires RGB, strip alpha
            image_b64 = image_to_base64(work.images.initial_image)
        if work.sampling:
            strength = work.sampling.denoise_strength
        noise_val = 0.0
        # Include mask for inpaint and refine_region
        if needs_mask and work.images and work.images.hires_mask:
            hires_mask = work.images.hires_mask
            # Diagnostic: check mask pixel values using Qt Image API
            log.warning(f"NAI MASK DIAG: hires_mask extent={hires_mask.extent}, is_mask={hires_mask.is_mask}")
            avg = hires_mask.average()
            log.warning(f"NAI MASK DIAG: mask average={avg:.4f} (0=all black, 1=all white)")
            # Sample some pixels
            w, h = hires_mask.extent
            center_px = hires_mask.pixel(w // 2, h // 2)
            corner_px = hires_mask.pixel(0, 0)
            log.warning(f"NAI MASK DIAG: center pixel={center_px}, corner pixel={corner_px}")
            # NAI requires feathered mask, like ComfyUI path (apply_grow_feather)
            if work.inpaint and (work.inpaint.grow > 0 or work.inpaint.feather > 0):
                hires_mask = _nai_feather_mask(hires_mask, work.inpaint.grow, work.inpaint.feather)
            mask_b64 = image_to_base64(hires_mask)

    if action is NaiAction.infill:
        if work.images and work.images.initial_image:
            # --- TEMP DIAGNOSTIC, REMOVE LATER — initial_image alpha channel check ---
            _init_img = work.images.initial_image
            _iiw, _iih = _init_img.extent
            _iiqimg = _init_img._qimage
            _ii_format = _iiqimg.format()
            log.warning(f"=== TEMP DIAGNOSTIC: INITIAL IMAGE ALPHA CHECK (REMOVE LATER) ===")
            log.warning(f"initial_image size: {_iiw}x{_iih}, QImage format: {_ii_format}")
            log.warning(f"initial_image is_mask: {_init_img.is_mask}, is_rgba: {_init_img.is_rgba}")
            if _init_img.is_rgba:
                # Sample alpha at corners and center
                from PyQt5.QtGui import qAlpha as _qAlpha
                _corners = [(0,0), (_iiw-1,0), (0,_iih-1), (_iiw-1,_iih-1), (_iiw//2,_iih//2)]
                _alphas = []
                for _cx, _cy in _corners:
                    _alphas.append(_qAlpha(_iiqimg.pixel(_cx, _cy)))
                log.warning(f"alpha samples (corners+center): {_alphas}")
                # Check a row of alpha in the mask white region (if known)
                _all_opaque = all(a == 255 for a in _alphas)
                _all_transparent = all(a == 0 for a in _alphas)
                log.warning(f"all sampled alpha=255: {_all_opaque}, all=0: {_all_transparent}")
            # --- END TEMP DIAGNOSTIC ---
            work.images.initial_image.make_opaque()  # NAI requires RGB, strip alpha
            image_b64 = image_to_base64(work.images.initial_image)
        if work.images and work.images.hires_mask:
            # --- TEMP DIAGNOSTIC, REMOVE LATER — mask pixel distribution + geometry ---
            _mask_img = work.images.hires_mask
            _mw, _mh = _mask_img.extent
            _mqimg = _mask_img._qimage
            _pixel_data = bytearray()
            for _y in range(_mh):
                _ptr = _mqimg.scanLine(_y)
                if _ptr is not None:
                    _pixel_data.extend(_ptr.asstring(_mw))
            _total = len(_pixel_data)
            if _total > 0:
                _c0 = _pixel_data.count(0)
                _c255 = _pixel_data.count(255)
                _cmid = _total - _c0 - _c255
                _minv = min(_pixel_data)
                _maxv = max(_pixel_data)
                log.warning(f"=== TEMP DIAGNOSTIC: MASK PIXEL DISTRIBUTION (REMOVE LATER) ===")
                log.warning(f"mask size: {_mw}x{_mh}, total pixels: {_total}")
                log.warning(f"pure black (0): {_c0} ({_c0/_total*100:.2f}%)")
                log.warning(f"pure white (255): {_c255} ({_c255/_total*100:.2f}%)")
                log.warning(f"gray (1-254): {_cmid} ({_cmid/_total*100:.2f}%)")
                log.warning(f"min/max value: {_minv}/{_maxv}")

                # Image vs mask size comparison
                if work.images.initial_image:
                    _iw, _ih = work.images.initial_image.extent
                    _match = "MATCH" if (_iw == _mw and _ih == _mh) else "MISMATCH!"
                    log.warning(f"image size: {_iw}x{_ih}, mask size: {_mw}x{_mh} => {_match}")
                else:
                    log.warning(f"image size: N/A (no initial_image), mask size: {_mw}x{_mh}")

                # White pixel bounding box (repaint region geometry)
                _x1, _y1, _x2, _y2 = _mw, _mh, 0, 0
                for _y in range(_mh):
                    for _x in range(_mw):
                        if _pixel_data[_y * _mw + _x] > 127:
                            _x1 = min(_x1, _x)
                            _y1 = min(_y1, _y)
                            _x2 = max(_x2, _x)
                            _y2 = max(_y2, _y)
                if _x2 >= _x1:
                    _bw = _x2 - _x1 + 1
                    _bh = _y2 - _y1 + 1
                    _area_pct = (_bw * _bh) / _total * 100
                    log.warning(f"white bbox: ({_x1},{_y1})-({_x2},{_y2}), size {_bw}x{_bh}, area {_area_pct:.2f}% of image")
                else:
                    log.warning(f"white bbox: NO WHITE PIXELS FOUND")

                # Mask polarity check: NAI expects white=repaint
                log.warning(f"mask polarity: is_mask={_mask_img.is_mask}, format={_mqimg.format()}")
                log.warning(f"NAI convention: white(255)=repaint, black(0)=keep. Verify visually.")
            # --- END TEMP DIAGNOSTIC ---
            # NAI requires feathered mask, like ComfyUI path (apply_grow_feather)
            _mask_to_encode = work.images.hires_mask
            if work.inpaint and (work.inpaint.grow > 0 or work.inpaint.feather > 0):
                _mask_to_encode = _nai_feather_mask(_mask_to_encode, work.inpaint.grow, work.inpaint.feather)
            mask_b64 = image_to_base64(_mask_to_encode)
        if work.sampling:
            strength = work.sampling.denoise_strength

    # --- TEMP DIAGNOSTIC: STRENGTH PATH (REMOVE LATER) ---
    log.warning(f"TEMP DIAGNOSTIC: STRENGTH PATH [6] convert_workflow: strength after read={strength}, action={action}")
    # --- END TEMP DIAGNOSTIC ---
    # Clamp strength to 0.99 for img2img to avoid full replacement
    if action is NaiAction.img2img and strength >= 1.0:
        strength = 0.99

    # --- Vibe Transfer (from WorkflowInput control layers) ---
    ref_images: list[str] = []
    ref_strengths: list[float] = []
    ref_info_extracted: list[float] = []

    if cond:
        from .resources import ControlMode
        for ctrl in cond.control:
            if ctrl.mode in (ControlMode.reference, ControlMode.style) and ctrl.image is not None:
                ref_images.append(image_to_base64(ctrl.image))
                ref_strengths.append(ctrl.strength)
                ref_info_extracted.append(1.0)

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
        skip_cfg_above_sigma=skip_sigma,
        image=image_b64,
        strength=strength,
        noise=noise_val,
        mask=mask_b64,
        reference_image_multiple=ref_images if ref_images else None,
        reference_strength_multiple=ref_strengths if ref_strengths else None,
        reference_information_extracted_multiple=ref_info_extracted if ref_info_extracted else None,
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
