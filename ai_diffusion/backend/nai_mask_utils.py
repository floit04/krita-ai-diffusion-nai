"""NovelAI inpaint mask pipeline.

Port of the NAI launcher's `inpaint_mask/inpaint_mask_operations.dart`
(4.0.2, commit ae7990c), which itself mirrors NovelAI web build
ae6a6aa-production: mask input becomes coverage alpha, is downsampled to the
8px latent grid with canvas-like nearest sampling, thresholded at alpha > 155,
and re-expanded to the request size — that is the HTTP ``mask``. The
client-side composite mask is the same latent mask dilated 4 more iterations,
re-expanded, then blurred (web worker stack blur, radius 20 x2) so results can
be soft-blended without the server's add_original_image pass.

Environment constraints (Krita's embedded Python: stdlib + Qt5, no numpy):
- Per-pixel work stays in single-channel ``bytes``/``bytearray`` planes.
  The Dart code runs the same math on r/g/b of a grayscale image, so a single
  plane is byte-identical.
- The stack blur is an exact port, but restricted to the mask's bounding box
  plus a margin (outside is uniformly black in both views, so results are
  identical). Oversized regions fall back to a smooth scale approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter

from ..image import Extent, Image
from ..util import ensure

LATENT_GRID_SIZE = 8
LATENT_DILATION_ITERATIONS = 4
BLUR_RADIUS = 20
BLUR_ITERATIONS = 2
COVERAGE_THRESHOLD = 155

# Exact blur only up to this many region pixels; beyond it, approximate.
_EXACT_BLUR_LIMIT = 1_200_000

_THRESHOLD_LUT = bytes(1 if value > COVERAGE_THRESHOLD else 0 for value in range(256))


@dataclass
class InpaintMaskArtifacts:
    """What the request carries and what the client composites with."""

    request_mask: Image  # opaque black/white, request size — the HTTP `mask`
    composite_alpha: bytes  # soft mask, one byte per pixel, request size
    width: int
    height: int
    latent_width: int
    latent_height: int

    def composite_alpha_scaled(self, target: Extent) -> tuple[bytes, int, int]:
        """The composite mask resampled (smooth) to another size, e.g. the
        focused-inpaint crop. Launcher: Lanczos; here Qt smooth scaling."""
        if (self.width, self.height) == (target.width, target.height):
            return self.composite_alpha, self.width, self.height
        # QImage wraps the buffer without copying, so it must stay referenced
        # until after the scale reads it.
        buffer = self.composite_alpha
        qimg = QImage(buffer, self.width, self.height, self.width, QImage.Format.Format_Grayscale8)
        scaled = qimg.scaled(
            target.width,
            target.height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return _gray_bytes(scaled), target.width, target.height


def prepare_inpaint_mask_artifacts(
    mask: Image,
    target: Extent,
    closing_iterations: int = 0,
    expansion_iterations: int = 0,
    latent_grid_size: int = LATENT_GRID_SIZE,
    latent_dilation_iterations: int = LATENT_DILATION_ITERATIONS,
    blur_radius: int = BLUR_RADIUS,
    blur_iterations: int = BLUR_ITERATIONS,
) -> InpaintMaskArtifacts:
    assert target.width > 0 and target.height > 0 and latent_grid_size > 0
    latent_width = max(1, target.width // latent_grid_size)
    latent_height = max(1, target.height // latent_grid_size)

    coverage, cov_w, cov_h = _coverage_plane(mask)
    if closing_iterations > 0 or expansion_iterations > 0:
        # The launcher binarizes with isMaskedPixel (alpha > 8, max(rgb) >= 32)
        # before source-space morphology. Our coverage already folds alpha into
        # brightness, so the equivalent cut-off is value >= 32.
        binary = bytes(1 if v >= 32 else 0 for v in coverage)
        if closing_iterations > 0:
            binary = _dilate(binary, cov_w, cov_h, closing_iterations)
            binary = _erode(binary, cov_w, cov_h, closing_iterations)
        if expansion_iterations > 0:
            binary = _dilate(binary, cov_w, cov_h, expansion_iterations)
        coverage = bytes(255 if v else 0 for v in binary)

    sampled = _resize_nearest_canvas_like(coverage, cov_w, cov_h, latent_width, latent_height)
    latent_binary = sampled.translate(_THRESHOLD_LUT)

    request_plane = _expand_latent_plane(
        bytes(255 if v else 0 for v in latent_binary),
        latent_width,
        latent_height,
        latent_grid_size,
        target,
    )
    request_mask = _opaque_mask_image(request_plane, target)

    composite_binary = latent_binary
    if latent_dilation_iterations > 0:
        composite_binary = _dilate(
            composite_binary, latent_width, latent_height, latent_dilation_iterations
        )
    composite_plane = _expand_latent_plane(
        bytes(255 if v else 0 for v in composite_binary),
        latent_width,
        latent_height,
        latent_grid_size,
        target,
    )
    if blur_radius > 0 and blur_iterations > 0:
        composite_plane = _worker_blur(
            composite_plane, target.width, target.height, blur_radius, blur_iterations
        )

    return InpaintMaskArtifacts(
        request_mask=request_mask,
        composite_alpha=bytes(composite_plane),
        width=target.width,
        height=target.height,
        latent_width=latent_width,
        latent_height=latent_height,
    )


def apply_composite_mask(generated: Image, composite_alpha: bytes, extent: Extent) -> Image:
    """Transparent patch: RGB from the generated image, alpha multiplied by the
    composite mask (launcher applyCompositeMaskToGeneratedImage).

    Implemented with QPainter DestinationIn instead of a per-pixel loop; the
    only difference is that fully-masked-out pixels lose their (invisible) RGB.
    """
    assert generated.extent == extent
    result = generated._qimage.convertToFormat(QImage.Format.Format_ARGB32)
    alpha = QImage(
        composite_alpha, extent.width, extent.height, extent.width, QImage.Format.Format_Alpha8
    ).copy()  # detach from the Python buffer before painting with it
    painter = QPainter(result)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    painter.drawImage(0, 0, alpha)
    painter.end()
    return Image(result)


# ---------------------------------------------------------------------------
# Single-channel plane helpers
# ---------------------------------------------------------------------------


def _gray_bytes(qimage: QImage) -> bytes:
    """Grayscale8 pixel bytes with the 32-bit scanline padding stripped.

    Qt quietly promotes Grayscale8 to RGB32 in some operations (smooth
    scaling among them), so convert back first when needed.
    """
    if qimage.format() != QImage.Format.Format_Grayscale8:
        qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)
    width, height = qimage.width(), qimage.height()
    stride = qimage.bytesPerLine()
    bits = ensure(qimage.constBits(), "Accessing data of invalid image")
    bits.setsize(stride * height)
    data = bytes(bits)
    if stride == width:
        return data
    return b"".join(data[y * stride : y * stride + width] for y in range(height))


def _coverage_plane(mask: Image) -> tuple[bytes, int, int]:
    """Coverage alpha per pixel: (max(r,g,b) * a + 127) // 255.

    Our masks are opaque grayscale, for which Qt's Grayscale8 conversion is
    exactly max(r,g,b) (r==g==b) with a==255 — so the plane is the raw bytes.
    """
    qimg = mask._qimage
    if qimg.format() != QImage.Format.Format_Grayscale8:
        qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
    return _gray_bytes(qimg), qimg.width(), qimg.height()


def _resize_nearest_canvas_like(
    plane: bytes, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> bytes:
    """Canvas-drawImage-like nearest sampling: src = floor((i + 0.5) * s / d)."""
    if (src_w, src_h) == (dst_w, dst_h):
        return plane
    out = bytearray(dst_w * dst_h)
    xs = [min(src_w - 1, int((x + 0.5) * src_w / dst_w)) for x in range(dst_w)]
    for y in range(dst_h):
        sy = min(src_h - 1, int((y + 0.5) * src_h / dst_h))
        row = plane[sy * src_w : (sy + 1) * src_w]
        base = y * dst_w
        for x in range(dst_w):
            out[base + x] = row[xs[x]]
    return bytes(out)


def _expand_latent_plane(
    plane: bytes, latent_w: int, latent_h: int, grid: int, target: Extent
) -> bytes:
    """Integer x8 block upscale, then nearest to the target if it differs."""
    if grid > 1:
        rows = []
        for y in range(latent_h):
            row = plane[y * latent_w : (y + 1) * latent_w]
            expanded = b"".join(bytes([v]) * grid for v in row)
            rows.append(expanded * grid)
        plane = b"".join(rows)
        latent_w, latent_h = latent_w * grid, latent_h * grid
    if (latent_w, latent_h) != (target.width, target.height):
        plane = _resize_nearest_canvas_like(plane, latent_w, latent_h, target.width, target.height)
    return plane


def _opaque_mask_image(plane: bytes, extent: Extent) -> Image:
    """White-on-black opaque RGB image (what the HTTP mask field expects)."""
    gray = QImage(plane, extent.width, extent.height, extent.width, QImage.Format.Format_Grayscale8)
    return Image(gray.convertToFormat(QImage.Format.Format_RGB32).copy())


def _dilate(plane: bytes, width: int, height: int, iterations: int) -> bytes:
    current = plane
    for _ in range(iterations):
        out = bytearray(width * height)
        for y in range(height):
            y0, y1 = max(0, y - 1), min(height - 1, y + 1)
            base = y * width
            for x in range(width):
                x0, x1 = max(0, x - 1), min(width - 1, x + 1)
                masked = 0
                for ny in range(y0, y1 + 1):
                    row = ny * width
                    for nx in range(x0, x1 + 1):
                        if current[row + nx]:
                            masked = 1
                            break
                    if masked:
                        break
                out[base + x] = masked
        current = bytes(out)
    return current


def _erode(plane: bytes, width: int, height: int, iterations: int) -> bytes:
    """3x3 erosion; like the launcher, pixels on the image border erode away."""
    current = plane
    for _ in range(iterations):
        out = bytearray(width * height)
        for y in range(height):
            base = y * width
            if y == 0 or y == height - 1:
                continue
            for x in range(1, width - 1):
                masked = 1
                for ny in (y - 1, y, y + 1):
                    row = ny * width
                    if not (current[row + x - 1] and current[row + x] and current[row + x + 1]):
                        masked = 0
                        break
                out[base + x] = masked
        current = bytes(out)
    return current


# ---------------------------------------------------------------------------
# NovelAI web worker blur (stack-blur variant), exact port
# ---------------------------------------------------------------------------

# radius -> (mul, shg) from the canonical stackblur table; the NovelAI worker
# only ever calls radius 20.
_STACK_BLUR_CONSTANTS = {20: (39, 16)}


def _worker_blur(plane: bytes, width: int, height: int, radius: int, iterations: int) -> bytes:
    """Blur restricted to the mask's bounding box + margin (exact elsewhere:
    outside that region everything is black before and after)."""
    if radius < 1:
        return plane
    bbox = _plane_bbox(plane, width, height)
    if bbox is None:
        return plane
    margin = radius * max(1, min(iterations, 3)) + 1
    x0 = max(0, bbox[0] - margin)
    y0 = max(0, bbox[1] - margin)
    x1 = min(width, bbox[2] + margin)
    y1 = min(height, bbox[3] + margin)
    rw, rh = x1 - x0, y1 - y0

    region = bytearray(rw * rh)
    for y in range(rh):
        src = (y0 + y) * width + x0
        region[y * rw : (y + 1) * rw] = plane[src : src + rw]

    if rw * rh > _EXACT_BLUR_LIMIT or radius not in _STACK_BLUR_CONSTANTS:
        blurred = _approx_blur(bytes(region), rw, rh, radius, iterations)
    else:
        blurred = _stack_blur_exact(region, rw, rh, radius, iterations)

    out = bytearray(plane)
    for y in range(rh):
        dst = (y0 + y) * width + x0
        out[dst : dst + rw] = blurred[y * rw : (y + 1) * rw]
    return bytes(out)


def _plane_bbox(plane: bytes, width: int, height: int) -> tuple[int, int, int, int] | None:
    zero_row = bytes(width)
    top = bottom = None
    for y in range(height):
        if plane[y * width : (y + 1) * width] != zero_row:
            top = y
            break
    if top is None:
        return None
    for y in range(height - 1, top - 1, -1):
        if plane[y * width : (y + 1) * width] != zero_row:
            bottom = y
            break
    assert bottom is not None
    left, right = width, -1
    for y in range(top, bottom + 1):
        row = plane[y * width : (y + 1) * width]
        for x, v in enumerate(row):
            if v:
                left = min(left, x)
                break
        for x in range(len(row) - 1, -1, -1):
            if row[x]:
                right = max(right, x)
                break
    return left, top, right + 1, bottom + 1


def _stack_blur_exact(
    data: bytearray, width: int, height: int, radius: int, iterations: int
) -> bytearray:
    """Line-for-line port of the launcher's _officialWorkerBlur on one channel,
    including its quirks: the _hasAnyRgb guard on the horizontal running sum
    and forcing fully-black accumulators to stay black."""
    mul, shg = _STACK_BLUR_CONSTANTS[radius]
    right = width - 1
    bottom = height - 1
    radius_plus_one = radius + 1
    iteration_count = max(1, min(iterations, 3))
    sums = [0] * (width * height)
    min_x = [min(x + radius_plus_one, right) for x in range(width)]
    max_x = [max(x - radius, 0) for x in range(width)]
    min_y = [min(y + radius_plus_one, bottom) for y in range(height)]
    max_y = [max(y - radius, 0) for y in range(height)]
    cap_x = min(right, radius)

    for _ in range(iteration_count):
        flat = 0
        for y in range(height):
            row = y * width
            value = data[row] * radius_plus_one
            for x in range(1, cap_x + 1):
                value += data[row + x]
            if radius > right:
                value += data[row + right] * (radius - right)
            for x in range(width):
                sums[flat] = value
                add = data[row + min_x[x]]
                sub = data[row + max_x[x]]
                if add or sub:
                    value += add - sub
                flat += 1

        for x in range(width):
            flat_base = x
            value = sums[flat_base] * radius_plus_one
            for y in range(1, radius + 1):
                if y <= bottom:
                    flat_base += width
                value += sums[flat_base]
            offset = x
            for y in range(height):
                if value == 0:
                    data[offset] = 0
                else:
                    v = (value * mul) >> shg
                    data[offset] = min(v, 255)
                value += sums[x + min_y[y] * width] - sums[x + max_y[y] * width]
                offset += width
    return data


def _approx_blur(plane: bytes, width: int, height: int, radius: int, iterations: int) -> bytearray:
    """Fallback for oversized regions: smooth downscale + upscale approximates
    the blur natively. Kernel support chosen to match the stack blur's
    effective sigma (~radius * sqrt(iterations) / 2)."""
    factor = max(2, round(radius * math.sqrt(max(1, iterations)) / 2))
    qimg = QImage(plane, width, height, width, QImage.Format.Format_Grayscale8)
    small = qimg.scaled(
        max(1, width // factor),
        max(1, height // factor),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    back = small.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return bytearray(_gray_bytes(back))
