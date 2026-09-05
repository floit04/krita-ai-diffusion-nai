"""Focused inpaint geometry — NovelAI's "Focused Area Selection".

Normal NAI inpaint sends the whole canvas: it gets squashed to the target
resolution, processed, and stretched back. On a 4K canvas with a 720p target
the masked area is round-tripped through a handful of pixels and comes back a
level blurrier.

NAI's own answer is a focus box: the user drags a rectangle that caps the total
request area, paints the mask inside it, and only that box (plus a context
margin) is sent. This module resolves the geometry for that.

Direct port of `lib/core/utils/focused_inpaint_utils.dart` from
Aaalice_NAI_Launcher (NovelAI web build ae6a6aa-production, verified
2026-07-16). Names and structure deliberately mirror the Dart original so the
two can be diffed; the one intentional divergence is MAX_CROP_AREA (see below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from ..image import Bounds, Extent

DIMENSION_STEP = 64
FOCUSED_TARGET_AREA = 1_048_576  # 1024*1024 — free tier for Opus subscribers

# The reference uses maxRequestAreaPixels = 3_145_728 (the hard API cap) here.
# We deliberately cap at the free-tier area instead: the focus box then can never
# grow past 1 MP, so a focused request never costs Anlas. Because the constrained
# crop is always <= FOCUSED_TARGET_AREA, _resolve_target_size always takes the
# upscale branch and PRESERVE_CROP is effectively unreachable — it is kept only
# so the port stays faithful.
MAX_CROP_AREA = FOCUSED_TARGET_AREA

# Launcher 4.x: the web slider caps at 32~96 but accepts typed values, so the
# launcher widens the clamp to 32..192 with the web's default of 96.
MIN_CONTEXT_MIN = 32
MIN_CONTEXT_MAX = 192
MIN_CONTEXT_DEFAULT = 96

UPSCALE_TO_TARGET = "upscale_to_target"
PRESERVE_CROP = "preserve_crop"

_BINARY_SEARCH_ITERATIONS = 48


class Rect(NamedTuple):
    """Float rectangle, mirroring Dart's `Rect` (LTRB, may be inverted)."""

    left: float
    top: float
    right: float
    bottom: float

    @staticmethod
    def from_bounds(bounds: Bounds):
        return Rect(
            float(bounds.x),
            float(bounds.y),
            float(bounds.x + bounds.width),
            float(bounds.y + bounds.height),
        )

    @staticmethod
    def from_points(a: tuple[float, float], b: tuple[float, float]):
        return Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    @staticmethod
    def from_center(cx: float, cy: float, width: float, height: float):
        return Rect(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    @property
    def center(self):
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def intersect(self, other: Rect):
        # Dart's Rect.intersect does not normalize; an empty overlap yields an
        # inverted rect, which _resolve_selection_bounds then rejects.
        return Rect(
            max(self.left, other.left),
            max(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
        )


@dataclass
class FocusedGeometry:
    focus_bounds: Bounds
    """The focus box itself, clamped to the canvas (and shrunk if it was too big)."""

    context_crop: Bounds
    """focus_bounds padded by the minimum context margin — what actually gets sent."""

    request: Extent
    """Resolution the cropped image is sent at."""

    mode: str
    was_constrained: bool
    """True when the box had to be shrunk to respect MAX_CROP_AREA."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def resolve_geometry(
    source: Extent,
    focus: Bounds | Rect,
    min_context: int = MIN_CONTEXT_DEFAULT,
    fixed_anchor: tuple[float, float] | None = None,
) -> FocusedGeometry | None:
    """Resolve crop and request size for a focus box drawn on a `source`-sized canvas.

    `fixed_anchor` pins one corner while the opposite one is scaled back, which is
    what a corner drag needs. Returns None if the box does not overlap the canvas.
    """
    if source.width <= 0 or source.height <= 0:
        return None
    rect = focus if isinstance(focus, Rect) else Rect.from_bounds(focus)

    original_bounds = _resolve_selection_bounds(rect, source)
    if original_bounds is None:
        return None

    original_crop = _resolve_crop(original_bounds, source, min_context)
    must_constrain = original_crop.area > MAX_CROP_AREA
    constrained = (
        _constrain_to_max_area(rect, source, min_context, fixed_anchor) if must_constrain else rect
    )
    focus_bounds = _resolve_selection_bounds(constrained, source)
    if focus_bounds is None:
        return None

    return _resolve_geometry_from_bounds(focus_bounds, source, min_context, must_constrain)


def constrain_focus_bounds(
    source: Extent,
    focus: Bounds | Rect,
    min_context: int = MIN_CONTEXT_DEFAULT,
    fixed_anchor: tuple[float, float] | None = None,
) -> Bounds | None:
    """The focus box, shrunk if needed so its context crop fits MAX_CROP_AREA."""
    geometry = resolve_geometry(source, focus, min_context, fixed_anchor)
    return geometry.focus_bounds if geometry else None


def _resolve_geometry_from_bounds(
    focus_bounds: Bounds, source: Extent, min_context: int, was_constrained: bool
) -> FocusedGeometry:
    context_crop = _resolve_crop(focus_bounds, source, min_context)
    request, mode = _resolve_target_size(context_crop.extent)
    return FocusedGeometry(focus_bounds, context_crop, request, mode, was_constrained)


def _resolve_selection_bounds(rect: Rect, source: Extent) -> Bounds | None:
    left = _clamp(rect.left, 0.0, float(source.width))
    top = _clamp(rect.top, 0.0, float(source.height))
    right = _clamp(rect.right, left, float(source.width))
    bottom = _clamp(rect.bottom, top, float(source.height))

    width = round(right - left)
    height = round(bottom - top)
    if width <= 0 or height <= 0:
        return None
    return Bounds(math.floor(left), math.floor(top), width, height)


def _resolve_crop(bounds: Bounds, source: Extent, min_context: int) -> Bounds:
    # Named "minContextMegaPixels" upstream, but it is a plain pixel padding
    # applied on every side. Kept as-is rather than "fixed".
    padding = int(_clamp(min_context, MIN_CONTEXT_MIN, MIN_CONTEXT_MAX))
    return _expand_and_clamp(
        center_x=bounds.x + bounds.width / 2,
        center_y=bounds.y + bounds.height / 2,
        width=bounds.width + padding * 2,
        height=bounds.height + padding * 2,
        max_width=source.width,
        max_height=source.height,
    )


def _expand_and_clamp(
    center_x: float, center_y: float, width: int, height: int, max_width: int, max_height: int
) -> Bounds:
    resolved_width = int(_clamp(width, 1, max_width))
    resolved_height = int(_clamp(height, 1, max_height))
    x = math.floor(center_x - resolved_width / 2)
    y = math.floor(center_y - resolved_height / 2)
    x = int(_clamp(x, 0, max_width - resolved_width))
    y = int(_clamp(y, 0, max_height - resolved_height))
    return Bounds(x, y, resolved_width, resolved_height)


def _resolve_target_size(crop: Extent) -> tuple[Extent, str]:
    crop_area = crop.pixel_count
    mode = UPSCALE_TO_TARGET if crop_area <= FOCUSED_TARGET_AREA else PRESERVE_CROP
    scale = math.sqrt(FOCUSED_TARGET_AREA / crop_area) if mode == UPSCALE_TO_TARGET else 1.0

    width = _floor_to_grid(math.floor(crop.width * scale))
    height = _floor_to_grid(math.floor(crop.height * scale))
    area_limit = FOCUSED_TARGET_AREA if mode == UPSCALE_TO_TARGET else MAX_CROP_AREA

    if width * height > area_limit:
        if width >= height:
            width = _largest_grid_dimension_for_area(area_limit, height)
        else:
            height = _largest_grid_dimension_for_area(area_limit, width)

    return Extent(width, height), mode


def _floor_to_grid(value: int) -> int:
    return max(DIMENSION_STEP, (value // DIMENSION_STEP) * DIMENSION_STEP)


def _largest_grid_dimension_for_area(area_limit: int, other_dimension: int) -> int:
    grid_units = area_limit // other_dimension // DIMENSION_STEP
    return max(DIMENSION_STEP, grid_units * DIMENSION_STEP)


def _constrain_to_max_area(
    rect: Rect, source: Extent, min_context: int, fixed_anchor: tuple[float, float] | None
) -> Rect:
    """Binary search the largest scale of `rect` whose context crop fits MAX_CROP_AREA."""
    canvas = Rect(0.0, 0.0, float(source.width), float(source.height))
    clipped = rect.intersect(canvas)
    center = clipped.center

    anchor = None
    if fixed_anchor is not None:
        anchor = (
            _clamp(fixed_anchor[0], 0.0, float(source.width)),
            _clamp(fixed_anchor[1], 0.0, float(source.height)),
        )

    moving_corner = None
    if anchor is not None:
        moving_corner = (
            clipped.right
            if abs(anchor[0] - clipped.left) <= abs(anchor[0] - clipped.right)
            else clipped.left,
            clipped.bottom
            if abs(anchor[1] - clipped.top) <= abs(anchor[1] - clipped.bottom)
            else clipped.top,
        )

    def candidate_at(scale: float) -> Rect:
        if anchor is not None and moving_corner is not None:
            moving = (
                anchor[0] + (moving_corner[0] - anchor[0]) * scale,
                anchor[1] + (moving_corner[1] - anchor[1]) * scale,
            )
            return Rect.from_points(anchor, moving).intersect(canvas)
        return Rect.from_center(
            center[0],
            center[1],
            (clipped.right - clipped.left) * scale,
            (clipped.bottom - clipped.top) * scale,
        ).intersect(canvas)

    low, high = 0.0, 1.0
    best = _minimum_selection_rect(center, anchor, moving_corner, canvas)
    for _ in range(_BINARY_SEARCH_ITERATIONS):
        scale = (low + high) / 2
        candidate = candidate_at(scale)
        bounds = _resolve_selection_bounds(candidate, source)
        if bounds is None:
            low = scale
            continue
        if _resolve_crop(bounds, source, min_context).area <= MAX_CROP_AREA:
            low = scale
            best = candidate
        else:
            high = scale
    return best


def _minimum_selection_rect(
    center: tuple[float, float],
    anchor: tuple[float, float] | None,
    moving_corner: tuple[float, float] | None,
    canvas: Rect,
) -> Rect:
    if anchor is not None and moving_corner is not None:
        x_direction = 1.0 if moving_corner[0] >= anchor[0] else -1.0
        y_direction = 1.0 if moving_corner[1] >= anchor[1] else -1.0
        return Rect.from_points(
            anchor, (anchor[0] + 3 * x_direction, anchor[1] + 3 * y_direction)
        ).intersect(canvas)
    return Rect.from_center(center[0], center[1], 3, 3).intersect(canvas)


# -- Capping a dragged box ---------------------------------------------------
# The reference implementation only ever scales an oversized rectangle down about
# its centre, which reads as "the box jumped back to a smaller one". What NovelAI's
# own editor does, and what the box needs to do here, is refuse to grow: the side
# being pulled keeps the size it was pulled to and the other side gives way to stay
# inside the area budget.

MIN_FOCUS_SIDE = 64
"""No side is ever capped below one generation grid step."""


def _crop_span(focus_span: int, padding: int, source_span: int) -> int:
    """Context crop span for one axis — same clamping as _expand_and_clamp."""
    return int(_clamp(focus_span + 2 * padding, 1, source_span))


def _crop_area(width: int, height: int, padding: int, source: Extent) -> int:
    return _crop_span(width, padding, source.width) * _crop_span(height, padding, source.height)


def _max_focus_span(other_crop_span: int, padding: int, source_span: int) -> int:
    """Largest focus span on one axis, given the crop span already used by the other."""
    if other_crop_span <= 0:
        return source_span
    crop_limit = MAX_CROP_AREA // other_crop_span
    if crop_limit >= source_span:
        return source_span  # padding is clipped by the canvas, so the box may fill it
    return int(_clamp(crop_limit - 2 * padding, 0, source_span))


def _fit_driven(
    width: int, height: int, padding: int, source: Extent, drive_width: bool
) -> tuple[int, int]:
    """Keep the side the user is dragging; let the other one give way."""
    if drive_width:
        crop_width = _crop_span(width, padding, source.width)
        height = min(height, _max_focus_span(crop_width, padding, source.height))
        if height < MIN_FOCUS_SIDE:
            # The drag is past what any height can pay for: floor the height and
            # cap the dragged side instead, so it simply stops following the mouse.
            height = min(MIN_FOCUS_SIDE, source.height)
            crop_height = _crop_span(height, padding, source.height)
            width = min(width, _max_focus_span(crop_height, padding, source.width))
    else:
        crop_height = _crop_span(height, padding, source.height)
        width = min(width, _max_focus_span(crop_height, padding, source.width))
        if width < MIN_FOCUS_SIDE:
            width = min(MIN_FOCUS_SIDE, source.width)
            crop_width = _crop_span(width, padding, source.width)
            height = min(height, _max_focus_span(crop_width, padding, source.height))
    return max(1, width), max(1, height)


def _fit_uniform(width: int, height: int, padding: int, source: Extent) -> tuple[int, int]:
    """Both sides at once — the only sensible reading when there is nothing to diff against."""
    low, high = 0.0, 1.0
    best = (max(1, min(width, MIN_FOCUS_SIDE)), max(1, min(height, MIN_FOCUS_SIDE)))
    for _ in range(_BINARY_SEARCH_ITERATIONS):
        scale = (low + high) / 2
        candidate = (max(1, int(width * scale)), max(1, int(height * scale)))
        if _crop_area(*candidate, padding, source) <= MAX_CROP_AREA:
            low = scale
            best = candidate
        else:
            high = scale
    return best


def _anchor_capped(
    bounds: Bounds, previous: Bounds | None, width: int, height: int, source: Extent
) -> tuple[int, int]:
    """Pin the edges the user is not dragging, so the box grows out of one corner."""
    left_fixed = top_fixed = True
    if previous is not None:
        left_fixed = abs(bounds.x - previous.x) <= abs(
            (bounds.x + bounds.width) - (previous.x + previous.width)
        )
        top_fixed = abs(bounds.y - previous.y) <= abs(
            (bounds.y + bounds.height) - (previous.y + previous.height)
        )
    x = bounds.x if left_fixed else bounds.x + bounds.width - width
    y = bounds.y if top_fixed else bounds.y + bounds.height - height
    return (
        int(_clamp(x, 0, max(0, source.width - width))),
        int(_clamp(y, 0, max(0, source.height - height))),
    )


def cap_focus_box(
    source: Extent,
    requested: Bounds,
    previous: Bounds | None = None,
    min_context: int = MIN_CONTEXT_DEFAULT,
) -> Bounds | None:
    """Cap a dragged focus box at MAX_CROP_AREA without resetting it.

    `previous` is the last accepted box; diffing against it is what says which side
    is being dragged. Pull the width out at full height and the height shrinks to
    pay for it; pull further than any height can pay for and the width stops
    following the mouse. Without a `previous` (first read, or a box restored from
    file) both sides scale together, matching the reference implementation.

    Returns the box to use, or None if it does not overlap the canvas.
    """
    bounds = _resolve_selection_bounds(Rect.from_bounds(requested), source)
    if bounds is None:
        return None
    padding = int(_clamp(min_context, MIN_CONTEXT_MIN, MIN_CONTEXT_MAX))
    width, height = bounds.width, bounds.height
    if _crop_area(width, height, padding, source) <= MAX_CROP_AREA:
        return bounds

    grew_width = previous is not None and width > previous.width
    grew_height = previous is not None and height > previous.height
    if grew_width and not grew_height:
        width, height = _fit_driven(width, height, padding, source, drive_width=True)
    elif grew_height and not grew_width:
        width, height = _fit_driven(width, height, padding, source, drive_width=False)
    else:
        width, height = _fit_uniform(width, height, padding, source)

    x, y = _anchor_capped(bounds, previous, width, height, source)
    return Bounds(x, y, width, height)


def boxes_match(a: Bounds, b: Bounds, tolerance: int = 2) -> bool:
    """True when two boxes differ by no more than point/pixel rounding.

    Shape geometry round-trips through points, so a box written to the canvas can
    read back a pixel off. Without this, that pixel would look like a drag.
    """
    return (
        abs(a.x - b.x) <= tolerance
        and abs(a.y - b.y) <= tolerance
        and abs(a.width - b.width) <= tolerance
        and abs(a.height - b.height) <= tolerance
    )
