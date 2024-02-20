# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
import evaluate
from loguru import logger
from datasets import load_dataset
from models.generation_utils import get_logits_processor
import ttnn
import tt_lib

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from models.experimental.functional_t5.tt import ttnn_functional_t5
from models.experimental.functional_t5.tt import ttnn_optimized_functional_t5


from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        context = []
        question = []
        for i in range(batch):
            context.append(input_data[i]["context"])
            question.append(input_data[i]["question"])

        return context, question


def run_generate(input_ids, model, config, parameters, device, max_tokens, batch_size, use_optimized_version=False):
    decoded_tt_output = []

    logits_processor = get_logits_processor(input_ids, config)

    decoder_start_values = model.generation_config.pad_token_id * torch.ones(1, 128).to(torch.long)
    decoder_input_ids = model.generation_config.pad_token_id * torch.ones(batch_size, input_ids.shape[-1]).to(
        torch.long
    )

    input_ids = ttnn.from_torch(input_ids)
    input_ids = ttnn.to_device(input_ids, device)

    for iteration in range(max_tokens):
        decoder_input_ids = ttnn.from_torch(decoder_input_ids)
        decoder_input_ids = ttnn.to_device(decoder_input_ids, device)

        tt_model = ttnn_optimized_functional_t5 if use_optimized_version else ttnn_functional_t5

        tt_output, encoder_hidden_states = tt_model.t5_for_conditional_generation(
            config,
            input_ids,
            decoder_input_ids,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        next_token_logits = ttnn.to_torch(tt_output)

        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        decoder_input_ids = ttnn.from_device(decoder_input_ids)
        decoder_input_ids = ttnn.to_torch(decoder_input_ids)

        if (iteration + 1) % 32 == 0:
            decoder_input_ids = torch.cat([decoder_input_ids, decoder_start_values], dim=1)

        decoder_input_ids[:, iteration + 1] = next_tokens[:, iteration]

    return decoder_input_ids


def run_functional_t5_question_and_answering_inference(
    device, batch_size, sequence_length, max_tokens, model_name, input_path, use_optimized_version
):
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    context, question = load_inputs(input_path, batch_size)

    input_sentance = [f"question: {q} context: {c}" for q, c in zip(question, context)]

    profiler.start(f"preprocessing_input")
    input_ids = tokenizer(
        input_sentance,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    profiler.end(f"preprocessing_input")

    tt_model_name = "ttnn_" + ("optimized_" if use_optimized_version else "") + model_name

    decoded_tt_output = []

    converter = ttnn_optimized_functional_t5.converter if use_optimized_version else ttnn_functional_t5.converter

    profiler.start(f"preprocessing_parameter")
    parameters = ttnn.model_converter.from_torch_model(
        cache_name=tt_model_name,
        model=lambda: model,
        converter=converter,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    profiler.start(f"inference_time")
    tt_output = run_generate(
        input_ids,
        model,
        config,
        parameters,
        device,
        max_tokens,
        batch_size,
        use_optimized_version,
    )
    profiler.end(f"inference_time")

    profiler.start(f"post_processing_output_to_string")
    for batch in range(batch_size):
        output = tokenizer.decode(tt_output[batch], skip_special_tokens=True)
        decoded_tt_output.append(output)
    profiler.end(f"post_processing_output_to_string")

    logger.info(decoded_tt_output)

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


def run_functional_t5_question_and_answering_inference_squadv2(
    device, batch_size, sequence_length, max_tokens, model_name, use_optimized_version
):
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    squad_dataset = load_dataset("squad_v2")
    validation_split = squad_dataset["validation"]
    predicted_answers = []
    reference_answer = []
    decoded_tt_output = []

    tt_model_name = "ttnn_" + ("optimized_" if use_optimized_version else "") + model_name
    converter = ttnn_optimized_functional_t5.converter if use_optimized_version else ttnn_functional_t5.converter

    parameters = ttnn.model_converter.from_torch_model(
        model_name=tt_model_name,
        model=lambda: model,
        converter=converter,
        device=device,
    )

    question = []
    context = []
    answers = []
    id = []

    index = 0
    while index < batch_size:
        answer = validation_split["answers"][index]
        if len(answer["text"]) > 0:
            question.append(validation_split["question"][index])
            context.append(validation_split["context"][index])
            answers.append(validation_split["answers"][index])
            id.append(validation_split["id"][index])
            index += 1
        else:
            continue

    input_sentance = [f"question: {q} context: {c}" for q, c in zip(question, context)]

    input_ids = tokenizer(
        input_sentance,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    tt_output = run_generate(
        input_ids,
        model,
        config,
        parameters,
        device,
        max_tokens,
        batch_size,
        use_optimized_version,
    )

    for batch in range(batch_size):
        output = tokenizer.decode(tt_output[batch], skip_special_tokens=True)
        decoded_tt_output.append(output)

    logger.info(decoded_tt_output)

    for batch in range(batch_size):
        predicted_answers.append(
            {
                "prediction_text": decoded_tt_output[batch],
                "id": id[batch],
                "no_answer_probability": 0.0,
            }
        )
        reference_answer.append(
            {
                "answers": {
                    "answer_start": [answers[batch]["answer_start"][0]],
                    "text": [answers[batch]["text"][0]],
                },
                "id": id[batch],
            }
        )
    squad_metric = evaluate.load("squad_v2")
    eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)
    logger.info("Exact Match :")
    logger.info(eval_score["exact"])
    logger.info("F1 Score :")
    logger.info(eval_score["f1"])


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name", "use_optimized_version"),
    (
        (8, 384, 5, "t5-small", False),
        (8, 384, 5, "google/flan-t5-small", False),
    ),
)
def test_functional_t5_demo(
    device, batch_size, sequence_length, max_tokens, model_name, input_path, use_optimized_version
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_functional_t5_question_and_answering_inference(
        device, batch_size, sequence_length, max_tokens, model_name, input_path, use_optimized_version
    )


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name", "use_optimized_version"),
    ((3, 384, 5, "t5-small", False), (3, 384, 5, "google/flan-t5-small", False)),
)
def test_functional_t5_demo_squadv2(device, batch_size, sequence_length, max_tokens, model_name, use_optimized_version):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_functional_t5_question_and_answering_inference_squadv2(
        device, batch_size, sequence_length, max_tokens, model_name, use_optimized_version
    )
