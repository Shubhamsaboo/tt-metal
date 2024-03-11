# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import transformers


import ttnn

from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert
from models.demos.bert.tt import ttnn_optimized_sharded_bert

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(bert):
    return {
        ttnn_bert: (15, 32),
        ttnn_optimized_bert: (12, 0.08),
        ttnn_optimized_sharded_bert: (12, 0.08),
    }[bert]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
# @pytest.mark.parametrize("bert", [ttnn_bert, ttnn_optimized_bert, ttnn_optimized_sharded_bert])
@pytest.mark.parametrize("bert", [ttnn_optimized_sharded_bert])
def test_performance(device, use_program_cache, model_name, batch_size, sequence_size, bert):
    # disable_persistent_kernel_cache()

    config = transformers.BertConfig.from_pretrained(model_name)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if bert != ttnn_bert else None

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    elif bert == ttnn_optimized_sharded_bert:
        tt_model_name = f"ttnn_{model_name}_optimized_sharded"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    durations = []
    for _ in range(1):
        ttnn_bert_inputs = bert.preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            torch_attention_mask,
            device=device,
        )

        start = time.time()
        with ttnn.disable_validate_decorator():
            tt_output = bert.bert_for_question_answering(
                config,
                *ttnn_bert_inputs,
                parameters=parameters,
            )
            tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        # enable_persistent_kernel_cache()

    # inference_and_compile_time, inference_time, *_ = durations
    inference_time, *_ = durations
    """
    expected_compile_time, expected_inference_time = get_expected_times(bert)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )
    """
    # logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    # logger.info(f"Samples per second: {1 / inference_time * batch_size}")
