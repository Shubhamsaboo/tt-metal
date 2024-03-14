# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from datasets import load_dataset
from loguru import logger
import ttnn
from transformers import (
    AutoFeatureExtractor,
    WhisperModel,
    WhisperConfig,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from models.generation_utils import get_logits_processor
from ttnn.model_preprocessing import preprocess_model_parameters

import torch

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_dataset


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_generate(
    config,
    input_embeds,
    input_features,
    ttnn_model,
    decoder_hidden_states,
    decoder_attention_mask,
    parameters,
    processor,
    ttnn_linear_weight,
    device,
):
    input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id

    logits_processor = get_logits_processor(input_ids, config)

    input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)

    for i in range(32):
        output = ttnn_model.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
        )
        output = output @ ttnn_linear_weight

        output = ttnn.from_device(output)

        logits_to_torch = ttnn.to_torch(output)

        next_token_logits = logits_to_torch[:, i, :]

        next_tokens_scores = logits_processor(input_features, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        input_ids[:, i + 1] = next_tokens[:, None]

        decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_decoder_inputs(
            config=config, input_ids=input_ids, attention_mask=None, parameters=parameters.decoder, device=device
        )

        if next_tokens == config.eos_token_id:
            break
        logger.info(processor.batch_decode(input_ids, skip_special_tokens=True)[0])

    ttnn_transcription = processor.batch_decode(input_ids, skip_special_tokens=True)[0]

    return ttnn_transcription


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
def test_demo_functional_whisper_for_audio_classification(ttnn_model, device):
    torch.manual_seed(1234)

    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    )

    input_features = inputs.input_features

    logger.debug("Input audio language:")
    logger.debug(sample["language"])

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    config = model.config
    input_embedding = ttnn_model.preprocess_encoder_inputs(
        input_features=input_features, parameters=parameters.encoder, device=device
    )

    encoder_outputs = ttnn_model.encoder(config=config, inputs_embeds=input_embedding, parameters=parameters.encoder)

    hidden_states = ttnn.matmul(encoder_outputs, parameters.projector.weight)
    hidden_states = ttnn.add(hidden_states, parameters.projector.bias)

    pooled_output = ttnn.mean(hidden_states, dim=-2, keepdim=True)

    logits = ttnn.matmul(pooled_output, parameters.classifier.weight)
    logits = ttnn.add(logits, parameters.classifier.bias)

    logits_torch = ttnn.to_torch(logits)
    predicted_class_ids = torch.argmax(logits_torch).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    logger.info("predicted_label")
    logger.info(predicted_label)


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
def test_demo_functional_whisper_for_conditional_generation(ttnn_model, device):
    torch.manual_seed(0)

    model = WhisperModel.from_pretrained("openai/whisper-tiny.en").to(torch.bfloat16).eval()

    config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    linear_weight = hf_reference_model.proj_out.weight

    linear_weight = hf_reference_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    dtype_to_use = torch.bfloat16
    input_features = inputs.input_features.type(dtype_to_use)

    decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
    decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)

    attention_mask = None

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        parameters=parameters,
        device=device,
    )

    ttnn_output = run_generate(
        config,
        input_embeds,
        input_features,
        ttnn_model,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
        processor=processor,
        ttnn_linear_weight=ttnn_linear_weight,
        device=device,
    )
    logger.info("Model Output")
    logger.info(ttnn_output)
