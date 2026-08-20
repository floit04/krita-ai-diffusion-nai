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
    convert_workflow,
    restore_nai_base_results,
    restore_nai_results,
)
from ai_diffusion.backend.nai_workflow import (
    NAI_EDIT_MAX_PIXELS,
    nai_auto_resolution,
    nai_edit_resolution,
    nai_resolution,
)
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.image import Extent, Image, ImageCollection


def _workflow(*control: ControlInput, checkpoint="nai-diffusion-4-5-full"):
    work = WorkflowInput(WorkflowKind.generate)
    work.images = ImageInput.from_extent(Extent(2048, 1536))
    work.models = CheckpointInput(checkpoint, Arch.nai)
    work.sampling = SamplingInput("euler_ancestral", "karras", 5.0, 28, seed=123)
    work.conditioning = ConditioningInput("test", control=list(control))
    return work


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
