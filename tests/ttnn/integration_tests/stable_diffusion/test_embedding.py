# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_embeddings import TtTimestepEmbedding


@skip_for_wormhole_b0()
def test_embeddings(
    device,
):
    torch.manual_seed(0)
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()
    time_embedding = model.time_embedding

    parameters = ttnn.model_converter.from_torch_model(model=lambda: time_embedding, device=device)

    input = torch.randn([1, 1, 2, 320])
    torch_output = time_embedding(input.squeeze(0).squeeze(0))

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = TtTimestepEmbedding(input, parameters=parameters)
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.squeeze(0).squeeze(0)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
