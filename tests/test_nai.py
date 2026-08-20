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
    convert_workflow,
    restore_nai_base_results,
    restore_nai_results,
)
from ai_diffusion.backend.nai_workflow import (
    NAI_EDIT_MAX_PIXELS,
    NaiAction,
    NaiModel,
    apply_quality_tags,
    build_generate_request,
    nai_auto_resolution,
    nai_edit_resolution,
    nai_resolution,
)
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.image import Extent, Image, ImageCollection
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


def test_v5_generate_request_uses_launch_contract_and_ignores_variety_plus():
    request = build_generate_request(
        "test",
        "bad",
        832,
        1216,
        model=NaiModel.v5_full,
        variety_plus=True,
    )
    params = request["parameters"]

    assert request["model"] == "nai-diffusion-5-full"
    assert params["params_version"] == 4
    assert params["noise_schedule"] == "karras"
    assert "skip_cfg_above_sigma" not in params
    assert params["use_coords"] is False
    assert params["legacy_uc"] is False


def test_v5_img2img_uses_selected_source_and_v5_request_contract():
    source = Image.create(Extent(320, 192))
    base = ControlInput(
        ControlMode.nai_base,
        source,
        strength=0.55,
        param2=0.15,
        target_extent=Extent(1344, 768),
    )

    request = convert_workflow(_workflow(base, checkpoint="nai-diffusion-5-full"))
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
    full = build_generate_request(
        "test", "bad", 832, 1216, model=NaiModel.v5_full, action=NaiAction.infill
    )
    curated = build_generate_request(
        "test", "bad", 832, 1216, model=NaiModel.v5_curated, action=NaiAction.infill
    )

    assert full["model"] == "nai-diffusion-5-full-inpainting"
    assert curated["model"] == "nai-diffusion-4-5-curated-inpainting"


def test_v5_quality_tags_match_the_launch_preset():
    expected = "subject, very aesthetic, masterpiece, no text"
    assert apply_quality_tags("subject", NaiModel.v5_curated) == expected
    assert apply_quality_tags("subject", NaiModel.v5_full) == expected


def test_v5_request_boundary_drops_vibe_and_precise_reference_fields():
    request = build_generate_request(
        "test",
        "bad",
        832,
        1216,
        model=NaiModel.v5_full,
        reference_image_multiple=["encoded-vibe"],
        reference_strength_multiple=[0.6],
        reference_information_extracted_multiple=[0.7],
        precise_references=[
            {
                "image": "reference",
                "caption": "character",
                "strength": 0.8,
                "secondary": 0.25,
            }
        ],
    )
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
    request = convert_workflow(work)
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

    params = convert_workflow(_workflow(base))["parameters"]

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

    request = convert_workflow(work)
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

    request = convert_workflow(work)
    params = request["parameters"]
    assert request["action"] == "infill"
    assert (params["width"], params["height"]) == (1728, 1792)
    assert params["width"] * params["height"] <= NAI_EDIT_MAX_PIXELS
    assert Image.from_base64(params["image"]).extent == Extent(1728, 1792)
    assert Image.from_base64(params["mask"]).extent == Extent(1728, 1792)


def test_precise_reference_v45_request_and_cost_fields():
    reference = Image.create(Extent(320, 512))
    precise = ControlInput(
        ControlMode.nai_precise_character,
        reference,
        strength=0.8,
        param2=0.75,
    )

    params = convert_workflow(_workflow(precise))["parameters"]

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

    v4_params = convert_workflow(_workflow(precise, checkpoint="nai-diffusion-4-full"))[
        "parameters"
    ]
    mixed_params = convert_workflow(_workflow(precise, vibe), vibe_encodings={1: "encoded-vibe"})[
        "parameters"
    ]

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

    params = convert_workflow(_workflow(vibe), vibe_encodings={0: "encoded-vibe"})["parameters"]

    assert params["reference_image_multiple"] == ["encoded-vibe"]
    assert params["reference_strength_multiple"] == [0.6]
    assert params["reference_information_extracted_multiple"] == [0.7]
