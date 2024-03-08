# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import math
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.tt import ttnn_optimized_vit
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(device, model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    torch_pixel_values = torch_random((batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
    )

    pixel_values = ttnn.from_torch(torch_pixel_values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit.vit_patch_embeddings(
        config,
        pixel_values,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    torch_output, *_ = model.vit.embeddings.patch_embeddings(torch_pixel_values)
    assert_with_pcc(torch_output, output[0], 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(device, model_name, batch_size, image_size, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    # torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
    )

    pixel_values = ttnn.from_torch(torch_pixel_values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit.vit_embeddings(
        config,
        pixel_values,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    torch_output, *_ = model.vit.embeddings(torch_pixel_values)
    # pad_size = (0, 0, 0, 27)
    # torch_output = torch.nn.functional.pad(torch_output, pad_size, "constant", 0)

    assert_with_pcc(torch_output, torch.squeeze(output, 0), 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(batch_size, 1, 1, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit.vit_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_intermediate(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit.vit_intermediate(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_output(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()

    torch_intermediate = torch_random((batch_size, sequence_size, config.intermediate_size), -1, 1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    intermediate = ttnn.from_torch(torch_intermediate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit.vit_output(
        config,
        intermediate,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_layer(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").vit.encoder.layer[0]

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(batch_size, 1, 1, sequence_size, dtype=torch.float32)

    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_optimized_vit.vit_layer(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])  ## padded from 197 to 224
def test_vit_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    ).vit.encoder

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output = model(torch_hidden_states, torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    output = ttnn_optimized_vit.vit_encoder(
        config,
        hidden_states,
        head_masks,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("image_channels", [3])
@pytest.mark.parametrize("sequence_size", [196])
def test_vit(device, model_name, batch_size, image_size, image_channels, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values).logits

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit.custom_preprocessor,
    )

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(torch_pixel_values, layout=ttnn.TILE_LAYOUT, device=device)

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    output = ttnn_optimized_vit.vit(
        config,
        pixel_values,
        head_masks,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0][0], 0.9999)