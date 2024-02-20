# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_cross_attention_down_block_2d import (
    cross_attention_down_block_2d,
)

from models.utility_functions import skip_for_wormhole_b0, tt_to_torch_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion.custom_preprocessing import converter


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, in_channels",
    [
        (
            2,
            320,
            32,
            32,
            0,
            320,
        ),
        (
            2,
            320,
            16,
            16,
            0,
            320,
        ),
        (
            2,
            640,
            8,
            8,
            2,
            1280,
        ),
    ],
)
def test_cross_attn_down_block_2d_256x256(device, model_name, N, C, H, W, index, in_channels):
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    down_block = pipe.unet.down_blocks[index]
    down_block.eval()
    state_dict = pipe.unet.state_dict()
    config = pipe.unet.config

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.randn(hidden_states_shape)

    encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    temb_shape = torch.Size([1, 1, 2, 1280])
    temb = torch.randn(temb_shape)

    attention_mask = None
    cross_attention_kwargs = None

    torch_output, torch_list_out = down_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    hidden_states = ttnn.to_device(hidden_states, device)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device)

    temb = ttnn.from_torch(temb, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device)

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: down_block,
        converter=converter,
        device=device,
    )

    ttnn_output, _ = cross_attention_down_block_2d(
        hidden_states,
        encoder_hidden_states,
        temb,
        in_channels=in_channels,
        out_channels=in_channels,
        attention_mask=None,
        add_downsample=True,
        cross_attention_kwargs={},
        config=config,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.98)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, in_channels",
    [
        (
            2,
            320,
            64,
            64,
            0,
            320,
        ),
        (
            2,
            320,
            32,
            32,
            0,
            320,
        ),
        (
            2,
            640,
            16,
            16,
            2,
            1280,
        ),
    ],
)
def test_cross_attn_down_block_2d_512x512(device, model_name, N, C, H, W, index, in_channels):
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    down_block = pipe.unet.down_blocks[index]
    down_block.eval()
    state_dict = pipe.unet.state_dict()
    config = pipe.unet.config

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.randn(hidden_states_shape)
    encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    temb_shape = torch.Size([1, 1, 2, 1280])
    temb = torch.randn(temb_shape)

    attention_mask = None
    cross_attention_kwargs = None

    torch_output, torch_list_out = down_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    hidden_states = ttnn.to_device(hidden_states, device)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device)

    temb = ttnn.from_torch(temb, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device)

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: down_block,
        converter=converter,
        device=device,
    )

    ttnn_output, _ = cross_attention_down_block_2d(
        hidden_states,
        encoder_hidden_states,
        temb,
        in_channels=in_channels,
        out_channels=in_channels,
        attention_mask=None,
        add_downsample=True,
        cross_attention_kwargs={},
        config=config,
        parameters=parameters,
        device=device,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.98)
