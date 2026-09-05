"""NovelAI request-image normalization.

Port of the launcher's `nai_resolution_adapter.dart` (4.0.2). The parts this
plugin uses: `normalize_image_for_request` (the launcher's
normalizeImageForRequest — an unconditional stretch to exactly the request
size, no letterboxing, pass-through when the size already matches) and
`find_official_import_resolution` (how NovelAI web picks an import size).

Divergence: the launcher resizes with its Pica Lanczos3 port. Krita's embedded
Python has no numpy and a pure-Python separable convolution is minutes-slow on
3 MP images, so this uses Qt's SmoothTransformation instead.
"""

from __future__ import annotations

from ..image import Extent, Image

# NovelAI web build ae6a6aa-production, verified 2026-07-16.
OFFICIAL_MAX_PIXELS = 3_145_728
OFFICIAL_NAI_TARGET_LONG_SIDE = 1216
OFFICIAL_NAI_TARGET_SHORT_SIDE = 896
OFFICIAL_GRID_SIZE = 64


def is_compatible(width: int, height: int) -> bool:
    return width % 64 == 0 and height % 64 == 0 and width >= 64 and height >= 64


def is_official_import_compatible(width: int, height: int) -> bool:
    return is_compatible(width, height) and width * height <= OFFICIAL_MAX_PIXELS


def _nearest_official_grid(value: float) -> int:
    return max(OFFICIAL_GRID_SIZE, round(value / OFFICIAL_GRID_SIZE) * OFFICIAL_GRID_SIZE)


def normalize_image_for_request(image: Image, target: Extent) -> Image:
    """Stretch to exactly the request size; untouched when it already matches."""
    if image.extent == target:
        return image
    return Image.scale(image, target)


def find_official_import_resolution(
    source_width: int,
    source_height: int,
    current_width: int | None = None,
    current_height: int | None = None,
) -> Extent:
    """NovelAI web's Image2Image import sizing: orient portrait, then pick the
    64-grid size on the 1216/896 base sides with the smaller aspect error."""
    source_aspect = source_width / source_height
    is_landscape = source_aspect > 1

    oriented_width, oriented_height = source_width, source_height
    if is_landscape:
        oriented_width, oriented_height = oriented_height, oriented_width

    oriented_aspect = oriented_width / oriented_height
    current_matches_aspect = (
        current_width is not None
        and current_height is not None
        and current_width / current_height == source_aspect
    )
    fits_current = (
        current_width is not None
        and current_height is not None
        and oriented_width <= current_width
        and oriented_height <= current_height
    )
    if current_matches_aspect and fits_current:
        assert current_width is not None and current_height is not None
        return Extent(current_width, current_height)

    if not is_official_import_compatible(oriented_width, oriented_height):
        width_from_long = _nearest_official_grid(OFFICIAL_NAI_TARGET_LONG_SIDE * oriented_aspect)
        height_from_short = _nearest_official_grid(OFFICIAL_NAI_TARGET_SHORT_SIDE / oriented_aspect)
        long_err = abs(width_from_long / OFFICIAL_NAI_TARGET_LONG_SIDE - oriented_aspect)
        short_err = abs(OFFICIAL_NAI_TARGET_SHORT_SIDE / height_from_short - oriented_aspect)
        if (
            long_err < short_err
            and width_from_long * OFFICIAL_NAI_TARGET_LONG_SIDE <= OFFICIAL_MAX_PIXELS
        ):
            oriented_width = width_from_long
            oriented_height = OFFICIAL_NAI_TARGET_LONG_SIDE
        elif OFFICIAL_NAI_TARGET_SHORT_SIDE * height_from_short <= OFFICIAL_MAX_PIXELS:
            oriented_width = OFFICIAL_NAI_TARGET_SHORT_SIDE
            oriented_height = height_from_short
        else:
            oriented_width = 512
            oriented_height = 512

    if is_landscape:
        oriented_width, oriented_height = oriented_height, oriented_width
    return Extent(oriented_width, oriented_height)
