# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.experimental.functional_vit.reference import torch_functional_vit
from models.experimental.functional_vit.tt import ttnn_functional_vit
from models.experimental.functional_vit.tt import ttnn_optimized_functional_vit

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("img_size_c", [3])
@pytest.mark.parametrize("img_size_h", [224])
@pytest.mark.parametrize("img_size_w", [224])
@pytest.mark.parametrize("functional_vit", [ttnn_functional_vit, ttnn_optimized_functional_vit])
def test_vit(device, use_program_cache, model_name, batch_size, sequence_size, functional_vit):
    torch.manual_seed(1234)

    config = transformers.BertConfig.from_pretrained(model_name)

    # TODO(arakhmati): re-enable the line below once the issue with ttnn.embedding is fixed
    # torch_vit_input = torch.randint(0, config.config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_vit_input = torch.randint(0, 1, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if functional_vit == ttnn_optimized_functional_vit else None

    torch_parameters = preprocess_model_parameters(
        f"torch_{model_name}",
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        convert_to_ttnn=lambda *_: False,
    )

    torch_output = torch_functional_vit.vit_for_question_answering(
        config,
        torch_vit_input,
        torch_token_type_ids,
        torch_attention_mask,
        parameters=torch_parameters,
    )

    if functional_vit == ttnn_functional_vit:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_vit == ttnn_optimized_functional_vit:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    ttnn_vit_inputs = functional_vit.preprocess_inputs(
        torch_vit_input,
        torch_token_type_ids,
        torch_attention_mask,
        device=device,
    )

    tt_output = functional_vit.vit_for_question_answering(
        config,
        *ttnn_vit_inputs,
        parameters=parameters,
    )
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output[..., :2]

    assert_with_pcc(torch_output, tt_output, 0.9999)
