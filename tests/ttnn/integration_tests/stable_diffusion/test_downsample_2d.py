# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torch import nn
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import torch_random, skip_for_wormhole_b0
from models.experimental.functional_stable_diffusion.custom_preprocessing import converter
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_downsample_2d import downsample_2d


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 320, 32, 32, 0),
        (2, 640, 16, 16, 1),
        (2, 1280, 8, 8, 2),
    ],
)
def test_downsample_2d_256x256(device, model_name, batch_size, in_channels, input_height, input_width, index):
    input_shape = batch_size, in_channels, input_height, input_width
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[index]
    ref_model = unet_downblock.downsamplers[0]

    parameters = ttnn.model_converter.from_torch_model(model=lambda: unet, converter=converter, device=device)
    parameters = parameters.down_blocks[index].downsamplers[0]

    torch_hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output = ref_model(torch_hidden_states)

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.ROW_MAJOR_LAYOUT
    )

    ttnn_output = downsample_2d(
        in_channels=in_channels,
        hidden_states=ttnn_hidden_state,
        device=device,
        parameters=parameters,
        use_conv=True,
        out_channels=in_channels,
        padding=1,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)))

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 320, 64, 64, 0),
        (2, 640, 32, 32, 1),
        (2, 1280, 16, 16, 2),
    ],
)
def test_downsample_2d_512x512(device, model_name, batch_size, in_channels, input_height, input_width, index):
    input_shape = batch_size, in_channels, input_height, input_width
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[index]
    ref_model = unet_downblock.downsamplers[0]

    parameters = ttnn.model_converter.from_torch_model(model=lambda: unet, converter=converter, device=device)
    parameters = parameters.down_blocks[index].downsamplers[0]

    torch_hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output = ref_model(torch_hidden_states)

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.ROW_MAJOR_LAYOUT
    )

    ttnn_output = downsample_2d(
        in_channels=in_channels,
        hidden_states=ttnn_hidden_state,
        device=device,
        parameters=parameters,
        use_conv=True,
        out_channels=in_channels,
        padding=1,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)))

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
