import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import math
import time
import torch
from torch import nn
import tt_lib
from loguru import logger
from python_api_testing.models.llama.llama_utils import tt2torch_tensor, torch2tt_tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama_split.hf_llama_classes import (
    TtLlamaModelFirstHFModel,
    TtLlamaModelSecondHFModel,
)

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from transformers.generation.configuration_utils import GenerationConfig

from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)


def _merge_criteria_processor_list(
    default_list,  # Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list,  # Union[LogitsProcessorList, StoppingCriteriaList],
):  # -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list

    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = (
                    "stopping criteria"
                    if isinstance(custom, StoppingCriteria)
                    else "logits processor"
                )
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `generate`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `generate` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list


def _get_logits_processor(
    generation_config: GenerationConfig,
    input_ids_seq_length: int,
    encoder_input_ids,  # torch.LongTensor
    prefix_allowed_tokens_fn,  # Callable[[int, torch.Tensor], List[int]],
    logits_processor,  # Optional[LogitsProcessorList]
):  # -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if (
        generation_config.diversity_penalty is not None
        and generation_config.diversity_penalty > 0.0
    ):
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )
    if (
        generation_config.encoder_repetition_penalty is not None
        and generation_config.encoder_repetition_penalty != 1.0
    ):
        processors.append(
            EncoderRepetitionPenaltyLogitsProcessor(
                penalty=generation_config.encoder_repetition_penalty,
                encoder_input_ids=encoder_input_ids,
            )
        )
    if (
        generation_config.repetition_penalty is not None
        and generation_config.repetition_penalty != 1.0
    ):
        processors.append(
            RepetitionPenaltyLogitsProcessor(
                penalty=generation_config.repetition_penalty
            )
        )
    if (
        generation_config.no_repeat_ngram_size is not None
        and generation_config.no_repeat_ngram_size > 0
    ):
        processors.append(
            NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size)
        )
    if (
        generation_config.encoder_no_repeat_ngram_size is not None
        and generation_config.encoder_no_repeat_ngram_size > 0
    ):
        if self.config.is_encoder_decoder:
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(
                    generation_config.encoder_no_repeat_ngram_size, encoder_input_ids
                )
            )
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if generation_config.bad_words_ids is not None:
        processors.append(
            NoBadWordsLogitsProcessor(
                generation_config.bad_words_ids, generation_config.eos_token_id
            )
        )
    if (
        generation_config.min_length is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_length > 0
    ):
        processors.append(
            MinLengthLogitsProcessor(
                generation_config.min_length, generation_config.eos_token_id
            )
        )
    if (
        generation_config.min_new_tokens is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length,
                generation_config.min_new_tokens,
                generation_config.eos_token_id,
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                generation_config.num_beams // generation_config.num_beam_groups,
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(
            ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id)
        )
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config.max_length, generation_config.forced_eos_token_id
            )
        )
    if generation_config.remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if generation_config.exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(
                generation_config.exponential_decay_length_penalty,
                generation_config.eos_token_id,
                input_ids_seq_length,
            )
        )
    if generation_config.suppress_tokens is not None:
        processors.append(
            SuppressTokensLogitsProcessor(generation_config.suppress_tokens)
        )
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = (
            begin_index
            if (
                input_ids_seq_length > 1
                or generation_config.forced_bos_token_id is None
            )
            else begin_index + 1
        )
        if generation_config.forced_decoder_ids is not None:
            # generation starts after the last token that is forced
            begin_index += generation_config.forced_decoder_ids[-1][0]
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index
            )
        )
    if generation_config.forced_decoder_ids is not None:
        processors.append(
            ForceTokensLogitsProcessor(generation_config.forced_decoder_ids)
        )
    processors = _merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if generation_config.renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors


def get_logits_processor(input_ids, config):
    generation_config = GenerationConfig.from_model_config(config)
    input_ids_seq_length = input_ids.shape[-1]

    logits_processor = _get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
    )

    return logits_processor


def pad_input_32_left(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([pad_tensor, tensor], dim=1)

    return tensor


def pad_input_32_right(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def gen_position_ids(input_ids):
    # get positions_ids values
    past_key_values_length = 0
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )

    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids


def get_next_llama_output_token(input_ids, out_tensor, order, model="Pytorch"):
    next_token_logits = out_tensor[:, -1, :]
    # pre-process distribution
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # argmax
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    logger.info(f"{model} forward {order+1}-th generated id: {next_tokens}")

    return next_tokens


def call_hf_llama_forward_func(
    hugging_face_reference_model,
    logits_processor,
    input_ids,
    attention_mask,
    num_words=2,
):
    # generate next words by using huggingface llama forward function
    for i in range(num_words):
        # pad input tensors
        input_ids_padded = pad_input_32_left(input_ids, configuration.pad_token_id)
        attention_mask_padded = pad_input_32_left(attention_mask, -1)
        position_ids_padded = gen_position_ids(input_ids_padded)

        pytorch_out = hugging_face_reference_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask_padded,
            position_ids=position_ids_padded,
        )

        # get next token
        next_tokens = get_next_llama_output_token(
            input_ids_padded, pytorch_out.logits, i
        )

        # update input ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.full((1, 1), 1)], dim=-1)
        position_ids = gen_position_ids(input_ids)


def call_tt_llama_forward_func(
    initial_prompt,
    logits_processor,
    input_ids,
    attention_mask,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    num_words=2,
):
    tt_generated_text = initial_prompt
    for i in range(num_words):
        # pad input tensors
        input_ids_padded = pad_input_32_left(input_ids, configuration.pad_token_id)
        attention_mask_padded = pad_input_32_left(
            attention_mask, configuration.pad_token_id
        )
        position_ids_padded = gen_position_ids(input_ids_padded)

        logger.debug(f"The first call started: loop {i+1}")
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        first_out = run_llama_split_inference(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start=first_decoder_start,
            num_decoders=num_consecutive_decoders,
            x_inputs=input_ids_padded,
            att_mask=attention_mask_padded,
            position_ids=position_ids_padded,
            half=1,
        )
        tt_lib.device.CloseDevice(device)
        logger.debug(f"The first call ended: loop {i+1}")

        # The second call -------------------------------------------------------
        logger.debug(f"The second call started: loop {i+1}")
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)

        # send input tensor from host to tt device
        tt_input = first_out

        tt_out = run_llama_split_inference(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start=second_decoder_start,
            num_decoders=num_consecutive_decoders,
            x_inputs=tt_input,
            att_mask=attention_mask_padded,
            position_ids=position_ids_padded,
            half=2,
        )
        logger.debug(f"The second call ended: loop {i+1}")

        # squeeze
        tt_out = tt_out.squeeze(1)

        # Get next token
        next_tokens = get_next_llama_output_token(
            input_ids_padded, tt_out, i, "Tenstorrent"
        )

        # save output words
        s = tokenizer.decode(next_tokens.item(), skip_special_tokens=True)
        logger.info(f"TT {i+1}-th generated word: {s}")
        tt_generated_text = tt_generated_text + " " + s

        # update input ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.full((1, 1), 1)], dim=-1)
        position_ids = gen_position_ids(input_ids)

        tt_lib.device.CloseDevice(device)
        device = None

    return tt_generated_text


def run_llama_split_inference(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    x_inputs=None,
    att_mask=None,
    position_ids=None,
    half=1,
):
    if half == 1:
        logger.info("First pass throught TT model")
        tt_llama_model = TtLlamaModelFirstHFModel(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )
    else:
        logger.info("Second pass throught TT model")
        tt_llama_model = TtLlamaModelSecondHFModel(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )

    # returned type from the model is tuple
    tt_output = tt2torch_tensor(tt_out[0])
    return tt_output


if __name__ == "__main__":
    torch.manual_seed(1234)

    # parameters --------------------------------------------------
    base_url = "model.layers"
    max_position_embeddings = 2048
    tokenizer_name = "huggyllama/llama-7b"
    llama_model_name = "huggyllama/llama-7b"
    # tokenizer_name = "hf-internal-testing/llama-tokenizer"
    # llama_model_name = "decapoda-research/llama-7b-hf"

    # how many decoders to use
    first_decoder_start = 0
    second_decoder_start = 16
    num_consecutive_decoders = 16

    # how many words to generate
    num_words = 10
    # parameters --------------------------------------------------

    # load llama pytorch model =======================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name
    )

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate real input ============================================================
    # prompt_1 = "I believe the meaning of life is"
    prompt = "Hey, are you conscious? Can you talk to me?"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    logger.info(f"Initial prompt: {prompt}")
    logger.debug(f"Initial prompt ids: {input_ids}")

    # get position_ids values
    seq_length = input_ids.shape[1]
    position_ids = gen_position_ids(input_ids)

    logits_processor = get_logits_processor(
        input_ids, hugging_face_reference_model.config
    )

    # generate Pytorch output of num_words with generate function ========================================================
    generate_ids = hugging_face_reference_model.generate(
        input_ids, logits_processor=logits_processor, max_length=seq_length + num_words
    )
    # logger.info(f"generate_ids shape: {generate_ids.shape}")
    output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    logger.info(f"PyTorch generated response: {output}")
    logger.debug(f"PyTorch generated ids: {generate_ids}")

    # generate Pytorch output: call forward() function several times ======================================================
    call_hf_llama_forward_func(
        hugging_face_reference_model,
        logits_processor,
        input_ids,
        attention_mask,
        num_words,
    )

    # TT output: call forward() function several times ====================================================================
    tt_generated_text = call_tt_llama_forward_func(
        prompt,
        logits_processor,
        input_ids,
        attention_mask,
        first_decoder_start,
        second_decoder_start,
        num_consecutive_decoders,
        num_words,
    )

    logger.info(f"PyTorch generated response: {output}")
    logger.info(f"Tenstorrent generated output: {tt_generated_text}")
