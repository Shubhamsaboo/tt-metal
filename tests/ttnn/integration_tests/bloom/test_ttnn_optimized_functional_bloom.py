# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from transformers.models import bloom

from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import torch_random, skip_for_wormhole_b0


from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.int64)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)

    torch_causal_mask = torch.empty((sequence_size, sequence_size), dtype=torch.bool)
    torch_seq_ids = torch.arange(sequence_size)
    torch_causal_mask[:, 0:] = torch_seq_ids[:, None] < torch_seq_ids[None, :]
    torch_causal_mask = torch_causal_mask[None, None, :, :].expand(
        batch_size, config.n_head, sequence_size, sequence_size
    )
    torch_causal_mask = torch_causal_mask.float()

    torch_output, *_ = model(
        torch_hidden_states,
        torch_residual,
        torch_alibi,
        torch_causal_mask,
    )

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: model,
        device=device,
        converter=ttnn_optimized_functional_bloom.converter,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_causal_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    alibi = ttnn_optimized_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch.bfloat16
    )
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_functional_bloom.bloom_attention(
        config,
        hidden_states,
        residual,
        alibi,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9956)
