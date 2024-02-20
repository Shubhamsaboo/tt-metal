# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from diffusers import UNet2DConditionModel
import ttnn


from models.experimental.functional_stable_diffusion.tt.ttnn_functional_feedforward import feedforward
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (
            1,
            2,
            1024,
            320,
            0,
        ),
        (
            1,
            2,
            256,
            640,
            1,
        ),
        (
            1,
            2,
            64,
            1280,
            2,
        ),
        (
            1,
            2,
            16,
            1280,
            2,
        ),
    ],
)
def test_feedforward_256x256(device, model_name, N, C, H, W, index, reset_seeds):
    input_shapes = (N, C, H, W)
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.down_blocks[index].attentions[0].transformer_blocks[0].ff
    config = model.config
    torch_hidden_states = torch_random(input_shapes, -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: ref_model,
        device=device,
    )

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    output = feedforward(
        config,
        ttnn_hidden_state,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (
            1,
            2,
            4096,
            320,
            3,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
        ),
    ],
)
def test_feedforward_512x512(device, model_name, N, C, H, W, index, reset_seeds):
    input_shapes = (N, C, H, W)
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.up_blocks[index].attentions[0].transformer_blocks[0].ff
    config = model.config
    torch_hidden_states = torch_random(input_shapes, -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: ref_model,
        device=device,
    )

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    output = feedforward(
        config,
        ttnn_hidden_state,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)
