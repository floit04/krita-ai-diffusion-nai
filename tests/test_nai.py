from unittest.mock import AsyncMock, patch

from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ControlInput,
    ImageInput,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.nai_client import (
    NaiClient,
    compose_infill_results,
    convert_workflow,
    restore_nai_base_results,
    restore_nai_results,
)
from ai_diffusion.backend.nai_focused import (
    FOCUSED_TARGET_AREA,
    MAX_CROP_AREA,
    MIN_CONTEXT_DEFAULT,
    UPSCALE_TO_TARGET,
    cap_focus_box,
    constrain_focus_bounds,
    resolve_geometry,
)
from ai_diffusion.backend.nai_registry import build_prompt_semantics
from ai_diffusion.backend.nai_request_builder import (
    ACTION_INFILL,
    NaiGenerationParams,
    PreciseReference,
    VibeReference,
    build_request,
)
from ai_diffusion.backend.nai_workflow import (
    NAI_EDIT_MAX_PIXELS,
    NaiModel,
    nai_auto_resolution,
    nai_edit_resolution,
    nai_resolution,
)
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.ui.control import _layer_list_change

from .conftest import qtapp


def test_control_layer_list_change_skips_stable_entries_and_updates_names_in_place():
    original = (
        ("canvas", "Canvas", "canvas"),
        ("layer", "Layer 1", "layer-1"),
        ("layer", "Layer 2", "layer-2"),
    )

    assert _layer_list_change(original, original) == "none"
    assert (
        _layer_list_change(
            original,
            (
                ("canvas", "Canvas", "canvas"),
                ("layer", "Renamed", "layer-1"),
                ("layer", "Layer 2", "layer-2"),
            ),
        )
        == "names"
    )
    assert (
        _layer_list_change(
            original,
            original + (("layer", "Layer 3", "layer-3"),),
        )
        == "structure"
    )


def _workflow(*control: ControlInput, checkpoint="nai-diffusion-4-5-full"):
    work = WorkflowInput(WorkflowKind.generate)
    work.images = ImageInput.from_extent(Extent(2048, 1536))
    work.models = CheckpointInput(checkpoint, Arch.nai)
    work.sampling = SamplingInput("euler_ancestral", "karras", 5.0, 28, seed=123)
    work.conditioning = ConditioningInput("test", control=list(control))
    return work


def test_v5_models_are_registered_and_curated_is_the_default():
    client = NaiClient(NaiClient.default_api_url, "test-token")

    assert NaiModel.default() is NaiModel.v5_curated
    assert NaiModel.list_generate()[:2] == [NaiModel.v5_curated, NaiModel.v5_full]
    assert NaiModel.list_display()[NaiModel.v5_curated] == "NAI Diffusion V5 (Curated)"
    assert NaiModel.v5_curated.value in client.models.checkpoints
    assert NaiModel.v5_full.value in client.models.checkpoints


def test_v5_generate_request_uses_launch_contract():
    request = build_request(
        NaiGenerationParams(model="nai-diffusion-5-full", width=832, height=1216, prompt="test")
    ).request_data
    params = request["parameters"]

    assert request["model"] == "nai-diffusion-5-full"
    assert request["use_new_shared_trial"] is True
    assert params["params_version"] == 4
    assert params["noise_schedule"] == "karras"
    assert params["use_coords"] is False
    assert params["legacy_uc"] is False
    # 4.x contract fields the old pipeline never sent
    assert params["ucPresetId"] == "none"
    assert params["qualityPresetId"] == "standard"
    assert params["image_format"] == "png"
    assert params["tag_hint_qt"] == 1
    assert params["tag_hint_uc_preset"] == 0
    assert params["autoSmea"] is False
    assert params["add_original_image"] is True
    assert params["straight_alpha"] is False
    # empty negative goes out as `uc`, not `negative_prompt`
    assert params["uc"] == "" and "negative_prompt" not in params
    # V5 hides Variety+ on the web; the launcher deliberately allows it, and
    # since retainsVarietyPlus is false there is no explicit null either.
    assert "skip_cfg_above_sigma" not in params


def test_v5_variety_plus_sends_the_scaled_sigma():
    params = build_request(
        NaiGenerationParams(
            model="nai-diffusion-5-full", width=832, height=1216, prompt="t", variety_plus=True
        )
    ).request_data["parameters"]
    assert abs(params["skip_cfg_above_sigma"] - 58.0) < 1e-9  # 832x1216 is the reference volume


def test_v5_img2img_uses_selected_source_and_v5_request_contract():
    source = Image.create(Extent(320, 192))
    base = ControlInput(
        ControlMode.nai_base,
        source,
        strength=0.55,
        param2=0.15,
        target_extent=Extent(1344, 768),
    )

    request = convert_workflow(_workflow(base, checkpoint="nai-diffusion-5-full")).request
    params = request["parameters"]

    assert request["action"] == "img2img"
    assert request["model"] == "nai-diffusion-5-full"
    assert params["params_version"] == 4
    assert params["noise_schedule"] == "karras"
    assert (params["width"], params["height"]) == (1344, 768)
    assert params["strength"] == 0.55
    assert params["noise"] == 0.15
    assert Image.from_base64(params["image"]).extent == Extent(1344, 768)


def test_v5_inpaint_routes_full_natively_and_curated_to_official_fallback():
    full = build_request(
        NaiGenerationParams(model="nai-diffusion-5-full", action=ACTION_INFILL, prompt="t")
    ).request_data
    curated = build_request(
        NaiGenerationParams(model="nai-diffusion-5-curated", action=ACTION_INFILL, prompt="t")
    ).request_data

    assert full["model"] == "nai-diffusion-5-full-inpainting"
    assert curated["model"] == "nai-diffusion-4-5-curated-inpainting"


def test_v5_quality_tags_match_the_launch_preset():
    expected = "subject, very aesthetic, masterpiece, no text"
    for model in ("nai-diffusion-5-curated", "nai-diffusion-5-full"):
        semantics = build_prompt_semantics("subject", "", model, quality_toggle=True, uc_preset=3)
        assert semantics.effective_prompt == expected


def test_v5_request_boundary_drops_vibe_and_precise_reference_fields():
    request = build_request(
        NaiGenerationParams(
            model="nai-diffusion-5-full",
            width=832,
            height=1216,
            prompt="test",
            vibes=[VibeReference("encoded-vibe", 0.6, 0.7)],
            precise_references=[PreciseReference("reference", "character", 0.8, 0.75)],
        )
    ).request_data
    params = request["parameters"]

    assert "reference_image_multiple" not in params
    assert "reference_strength_multiple" not in params
    assert "reference_information_extracted_multiple" not in params
    assert "director_reference_images" not in params
    assert "director_reference_descriptions" not in params


@qtapp
async def test_v5_vibe_controls_do_not_call_paid_encode_endpoint():
    vibe = ControlInput(
        ControlMode.nai_vibe,
        Image.create(Extent(64, 64)),
        strength=0.6,
        param2=0.7,
    )
    client = NaiClient(NaiClient.default_api_url, "test-token")

    with patch.object(client, "_post_binary", new_callable=AsyncMock) as post_binary:
        result = await client._ensure_vibe_encodings(
            _workflow(vibe, checkpoint="nai-diffusion-5-full")
        )

    assert result == {}
    post_binary.assert_not_awaited()


def test_nai_resolution_snaps_to_nearest_64_and_auto_scales_by_source_size():
    assert nai_resolution(Extent(1599, 914)) == Extent(1600, 896)
    assert nai_resolution(Extent(16, 9000)) == Extent(64, 2048)

    assert nai_auto_resolution(Extent(192, 192)) == Extent(1024, 1024)
    assert nai_auto_resolution(Extent(640, 360)) == Extent(1344, 768)
    assert nai_auto_resolution(Extent(1344, 768)) == Extent(1344, 768)
    assert nai_auto_resolution(Extent(1600, 896)) == Extent(1600, 896)
    assert nai_auto_resolution(Extent(2500, 1400)) == Extent(2048, 1152)
    assert nai_auto_resolution(Extent(3000, 3000)) == Extent(1728, 1792)


def test_nai_edit_resolution_never_exceeds_provider_pixel_limit():
    assert nai_edit_resolution(Extent(1792, 2048)) == Extent(1664, 1856)

    sources = [
        Extent(192, 192),
        Extent(1344, 768),
        Extent(1792, 2048),
        Extent(2048, 2048),
        Extent(8000, 1200),
    ]
    for source in sources:
        target = nai_auto_resolution(source)
        assert target.pixel_count <= NAI_EDIT_MAX_PIXELS
        assert target.width <= 2048
        assert target.height <= 2048
        assert target.width % 64 == 0
        assert target.height % 64 == 0


def test_img2img_base_controls_action_resolution_and_source():
    source = Image.create(Extent(192, 128))
    base = ControlInput(
        ControlMode.nai_base,
        source,
        strength=0.65,
        param2=0.1,
        target_extent=Extent(1216, 832),
    )
    work = _workflow(base)
    work.kind = WorkflowKind.inpaint
    request = convert_workflow(work).request
    params = request["parameters"]

    assert request["action"] == "img2img"
    assert request["model"] == "nai-diffusion-4-5-full"
    assert (params["width"], params["height"]) == (1216, 832)
    assert params["strength"] == 0.65
    assert params["noise"] == 0.1
    assert "mask" not in params
    assert "image" in params
    assert Image.from_base64(params["image"]).extent == Extent(1216, 832)


def test_img2img_request_boundary_clamps_an_oversized_saved_target():
    source = Image.create(Extent(1792, 2048))
    base = ControlInput(
        ControlMode.nai_base,
        source,
        target_extent=Extent(1792, 2048),
    )

    params = convert_workflow(_workflow(base)).request["parameters"]

    assert (params["width"], params["height"]) == (1664, 1856)
    assert params["width"] * params["height"] <= NAI_EDIT_MAX_PIXELS
    assert Image.from_base64(params["image"]).extent == Extent(1664, 1856)


def test_img2img_results_are_restored_before_history():
    source = Image.create(Extent(192, 128))
    base = ControlInput(
        ControlMode.nai_base,
        source,
        target_extent=Extent(1216, 832),
    )
    work = _workflow(base)
    provider_result = Image.create(Extent(1216, 832))

    restored = restore_nai_base_results(work, ImageCollection([provider_result]))

    assert len(restored) == 1
    assert restored[0].extent == source.extent


def test_redraw_uses_auto_target_and_restores_original_extent():
    source = Image.create(Extent(2500, 1400))
    work = _workflow()
    work.kind = WorkflowKind.refine
    work.images = ImageInput.from_extent(source.extent)
    work.images.initial_image = source
    work.nai_target_extent = nai_auto_resolution(source.extent)

    request = convert_workflow(work).request
    params = request["parameters"]
    assert request["action"] == "img2img"
    assert (params["width"], params["height"]) == (2048, 1152)
    assert Image.from_base64(params["image"]).extent == Extent(2048, 1152)

    provider_result = Image.create(Extent(2048, 1152))
    restored = restore_nai_results(work, ImageCollection([provider_result]))
    assert restored[0].extent == source.extent


def test_inpaint_above_provider_pixel_limit_is_reduced_to_a_legal_grid():
    source = Image.create(Extent(1800, 1800))
    work = _workflow()
    work.kind = WorkflowKind.inpaint
    work.images = ImageInput.from_extent(source.extent)
    work.images.initial_image = source
    work.images.hires_mask = Image.create(source.extent)
    work.nai_target_extent = nai_auto_resolution(source.extent)

    converted = convert_workflow(work)
    request = converted.request
    params = request["parameters"]
    assert request["action"] == "infill"
    assert (params["width"], params["height"]) == (1728, 1792)
    assert params["width"] * params["height"] <= NAI_EDIT_MAX_PIXELS
    assert Image.from_base64(params["image"]).extent == Extent(1728, 1792)
    assert Image.from_base64(params["mask"]).extent == Extent(1728, 1792)
    # Launcher infill contract: no server-side original blending (the client
    # composites with the soft mask instead), flat strength sent, nested
    # img2img carries color_correct below 100%.
    assert params["add_original_image"] is False
    assert "strength" in params
    assert converted.mask_artifacts is not None
    assert params["inpaintImg2ImgStrength"] == 1
    assert "img2img" not in params  # exactly 1.0 omits it, like the web UI


def test_inpaint_below_full_strength_sends_color_correct():
    source = Image.create(Extent(1024, 1024))
    work = _workflow()
    work.kind = WorkflowKind.inpaint
    work.sampling = SamplingInput("euler_ancestral", "karras", 5.0, 32, start_step=4, seed=1)
    work.images = ImageInput.from_extent(source.extent)
    work.images.initial_image = source
    work.images.hires_mask = Image.create(source.extent)
    work.nai_target_extent = Extent(1024, 1024)

    params = convert_workflow(work).request["parameters"]
    strength = 28 / 32  # (total 32 - start 4) / total 32
    assert abs(params["inpaintImg2ImgStrength"] - strength) < 1e-9
    assert params["img2img"] == {"strength": strength, "color_correct": True}


def test_precise_reference_v45_request_and_cost_fields():
    reference = Image.create(Extent(320, 512))
    precise = ControlInput(
        ControlMode.nai_precise_character,
        reference,
        strength=0.8,
        param2=0.75,
    )

    params = convert_workflow(_workflow(precise)).request["parameters"]

    assert params["director_reference_strength_values"] == [0.8]
    assert params["director_reference_secondary_strength_values"] == [0.25]
    assert params["director_reference_information_extracted"] == [1]
    assert params["director_reference_descriptions"] == [
        {
            "caption": {"base_caption": "character", "char_captions": []},
            "legacy_uc": False,
        }
    ]
    assert len(params["director_reference_images"]) == 1
    assert Image.from_base64(params["director_reference_images"][0]).extent == Extent(1024, 1536)


def test_precise_reference_is_dropped_for_v4_and_drops_vibe_when_active():
    image = Image.create(Extent(64, 64))
    precise = ControlInput(ControlMode.nai_precise_style, image)
    vibe = ControlInput(ControlMode.nai_vibe, image, strength=0.6, param2=0.7)

    v4_params = convert_workflow(_workflow(precise, checkpoint="nai-diffusion-4-full")).request[
        "parameters"
    ]
    mixed_params = convert_workflow(
        _workflow(precise, vibe), vibe_encodings={1: "encoded-vibe"}
    ).request["parameters"]

    assert "director_reference_images" not in v4_params
    assert "director_reference_images" in mixed_params
    assert "reference_image_multiple" not in mixed_params


def test_vibe_request_fields_when_precise_reference_is_absent():
    vibe = ControlInput(
        ControlMode.nai_vibe,
        Image.create(Extent(64, 64)),
        strength=0.6,
        param2=0.7,
    )

    params = convert_workflow(_workflow(vibe), vibe_encodings={0: "encoded-vibe"}).request[
        "parameters"
    ]

    assert params["reference_image_multiple"] == ["encoded-vibe"]
    assert params["reference_strength_multiple"] == [0.6]
    assert params["reference_information_extracted_multiple"] == [0.7]


# ---------------------------------------------------------------------------
# Focused inpaint geometry
# ---------------------------------------------------------------------------


def _geometry(source: Extent, focus: Bounds, min_context: int = 16):
    geometry = resolve_geometry(source, focus, min_context)
    assert geometry is not None
    return geometry


def test_focused_geometry_upscales_a_small_crop_to_the_one_megapixel_target():
    geometry = _geometry(Extent(512, 512), Bounds(0, 0, 512, 512))

    assert geometry.context_crop.area == 512 * 512
    assert geometry.mode == UPSCALE_TO_TARGET
    assert geometry.request == Extent(1024, 1024)


def test_focused_geometry_uses_the_upscale_branch_at_exactly_one_megapixel():
    geometry = _geometry(Extent(1024, 1024), Bounds(0, 0, 1024, 1024))

    assert geometry.context_crop.area == FOCUSED_TARGET_AREA
    assert geometry.mode == UPSCALE_TO_TARGET
    assert geometry.request == Extent(1024, 1024)
    assert not geometry.was_constrained


def test_focused_geometry_shrinks_the_box_just_above_the_one_megapixel_cap():
    """The reference allows up to 3,145,728 px here; we cap at the free tier instead."""
    geometry = _geometry(Extent(1025, 1024), Bounds(0, 0, 1025, 1024))

    assert geometry.was_constrained
    assert geometry.context_crop.area <= MAX_CROP_AREA


def test_focused_geometry_never_exceeds_the_free_tier_and_stays_on_the_64_grid():
    cases = [
        (Extent(2048, 1536), Bounds(0, 0, 2048, 1536)),
        (Extent(3072, 1024), Bounds(0, 0, 3072, 1024)),  # no per-side limit
        (Extent(3000, 500), Bounds(0, 0, 3000, 500)),
        (Extent(4096, 1850), Bounds(2000, 900, 120, 80)),
        (Extent(4096, 1850), Bounds(-500, -500, 6000, 6000)),  # box outside the canvas
    ]
    for source, focus in cases:
        geometry = _geometry(source, focus, min_context=56)
        assert geometry.context_crop.area <= MAX_CROP_AREA, (source, focus)
        assert geometry.request.pixel_count <= FOCUSED_TARGET_AREA, (source, focus)
        assert geometry.request.width % 64 == 0 and geometry.request.height % 64 == 0
        assert Bounds.clamp(geometry.context_crop, source) == geometry.context_crop


def test_focused_geometry_preserves_the_crop_aspect_ratio():
    geometry = _geometry(Extent(3072, 1024), Bounds(0, 0, 3072, 1024), min_context=16)
    crop, request = geometry.context_crop, geometry.request

    crop_ratio = crop.width / crop.height
    # Flooring both sides to the 64 grid is what moves the ratio, so the
    # tolerance has to be relative and roughly one grid step wide.
    assert abs(request.width / request.height - crop_ratio) / crop_ratio < 0.05


def test_focused_geometry_pads_the_box_by_the_minimum_context_on_every_side():
    focus = Bounds(2000, 900, 120, 80)
    geometry = _geometry(Extent(4096, 1850), focus, min_context=56)

    assert geometry.context_crop == Bounds(2000 - 56, 900 - 56, 120 + 112, 80 + 112)


def test_focused_geometry_clamps_the_minimum_context_to_the_supported_range():
    focus = Bounds(2000, 900, 120, 80)
    source = Extent(4096, 1850)

    assert _geometry(source, focus, 0).context_crop == _geometry(source, focus, 32).context_crop
    assert _geometry(source, focus, 9999).context_crop == _geometry(source, focus, 192).context_crop


def test_focused_geometry_rejects_a_box_that_misses_the_canvas():
    assert resolve_geometry(Extent(1024, 1024), Bounds(2000, 2000, 100, 100)) is None
    assert resolve_geometry(Extent(1024, 1024), Bounds(0, 0, 0, 0)) is None


def test_constrain_focus_bounds_keeps_a_legal_box_untouched():
    focus = Bounds(100, 100, 400, 400)
    assert constrain_focus_bounds(Extent(4096, 1850), focus, 56) == focus


def test_focused_inpaint_sends_only_the_crop_at_the_resolved_request_size():
    """The whole point: a small selection on a big canvas gets the full request.

    Without focus the 4096x1850 canvas would be squashed to the target resolution
    and the masked area would survive as a handful of pixels.
    """
    source = Extent(4096, 1850)
    geometry = resolve_geometry(source, Bounds(1200, 700, 120, 80), 56)
    assert geometry is not None

    work = _workflow()
    work.kind = WorkflowKind.inpaint
    work.images = ImageInput.from_extent(source)
    work.images.initial_image = Image.create(source)
    work.images.hires_mask = Image.create(source)
    work.nai_target_extent = geometry.request
    work.nai_focus_crop = geometry.context_crop

    request = convert_workflow(work).request
    params = request["parameters"]
    assert request["action"] == "infill"
    assert (params["width"], params["height"]) == (*geometry.request,)
    assert Image.from_base64(params["image"]).extent == geometry.request
    assert Image.from_base64(params["mask"]).extent == geometry.request
    # Free tier: the crop is capped at 1 MP, so a focused request never costs Anlas.
    assert geometry.request.pixel_count <= FOCUSED_TARGET_AREA


def test_focused_inpaint_results_land_back_at_the_crop_offset():
    source = Extent(4096, 1850)
    geometry = resolve_geometry(source, Bounds(1200, 700, 120, 80), 56)
    assert geometry is not None
    crop = geometry.context_crop

    work = _workflow()
    work.kind = WorkflowKind.inpaint
    work.images = ImageInput.from_extent(source)
    work.images.initial_image = Image.create(source)
    mask = Image.create(source, fill=0)
    mask.draw_image(Image.create(Bounds(0, 0, 120, 80).extent, fill=0xFFFFFFFF), (1200, 700))
    work.images.hires_mask = mask

    work.nai_target_extent = geometry.request
    work.nai_focus_crop = crop
    converted = convert_workflow(work)
    artifacts = converted.mask_artifacts
    assert artifacts is not None

    result = Image.create(geometry.request, fill=0xFF808080)
    restored = compose_infill_results(work, ImageCollection([result]), artifacts, crop)

    # Canvas-sized patch, so applying it needs no special case downstream.
    assert restored[0].extent == source
    # Opaque inside the mask, transparent everywhere else — including inside the
    # crop but outside the mask, and outside the crop entirely.
    assert restored[0].pixel(1260, 740)[3] == 255
    assert restored[0].pixel(crop.x + 2, crop.y + 2)[3] == 0
    assert restored[0].pixel(10, 10)[3] == 0


def _crop_area(box: Bounds, source: Extent, min_context: int = MIN_CONTEXT_DEFAULT) -> int:
    geometry = resolve_geometry(source, box, min_context)
    assert geometry is not None
    return geometry.context_crop.area


def test_dragging_a_focus_box_wider_shrinks_its_height_instead_of_resetting_it():
    # The box only ever gives way on the side the user is not dragging: pull the
    # width out and the height pays for it, so the box never jumps to a smaller one.
    source = Extent(4096, 1850)
    start = cap_focus_box(source, Bounds(0, 0, 500, 1850))
    assert start is not None and _crop_area(start, source) <= MAX_CROP_AREA

    wider = cap_focus_box(source, Bounds(0, 0, 900, start.height), start)
    assert wider is not None
    assert wider.width == 900  # the dragged side keeps exactly what was asked for
    assert wider.height < start.height
    assert _crop_area(wider, source) <= MAX_CROP_AREA

    widest = cap_focus_box(source, Bounds(0, 0, 4096, wider.height), wider)
    assert widest is not None
    assert widest.width == 4096 and widest.height < wider.height
    assert _crop_area(widest, source) <= MAX_CROP_AREA


def test_dragging_a_focus_box_taller_shrinks_its_width():
    source = Extent(4096, 1850)
    flat = cap_focus_box(source, Bounds(0, 0, 4096, 200))
    assert flat is not None

    taller = cap_focus_box(source, Bounds(0, 0, flat.width, 1850), flat)
    assert taller is not None
    assert taller.height == 1850 and taller.width < flat.width
    assert _crop_area(taller, source) <= MAX_CROP_AREA


def test_a_focus_box_that_already_fits_is_left_alone():
    source = Extent(4096, 1850)
    box = Bounds(100, 100, 512, 512)
    assert cap_focus_box(source, box, box) == box


def test_capping_a_focus_box_keeps_the_corner_that_is_not_being_dragged():
    # Dragging the right edge out must not walk the left edge across the canvas.
    source = Extent(4096, 1850)
    previous = Bounds(300, 200, 400, 1600)
    capped = cap_focus_box(source, Bounds(300, 200, 3000, 1600), previous)
    assert capped is not None
    assert (capped.x, capped.y) == (300, 200)
    assert _crop_area(capped, source) <= MAX_CROP_AREA


def test_a_focus_box_with_no_history_scales_both_sides_together():
    # Nothing to diff against (first read, or a box restored from a .kra file):
    # fall back to the reference implementation's proportional shrink.
    source = Extent(4096, 1850)
    capped = cap_focus_box(source, Bounds(0, 0, 4096, 1850))
    assert capped is not None
    assert capped.width < 4096 and capped.height < 1850
    assert _crop_area(capped, source) <= MAX_CROP_AREA


def test_capping_a_focus_box_respects_the_minimum_context_setting():
    source = Extent(4096, 1850)
    for min_context in (16, 56, 192):
        previous = Bounds(0, 0, 256, 256)
        capped = cap_focus_box(source, Bounds(0, 0, 4096, 256), previous, min_context)
        assert capped is not None
        assert _crop_area(capped, source, min_context) <= MAX_CROP_AREA
