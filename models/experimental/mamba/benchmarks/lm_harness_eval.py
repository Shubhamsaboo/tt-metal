import torch

import transformers
from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba-cpu-reference")
class MambaEvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained: MambaPretrainedModelName = "state-spaces/mamba-2.8b",
        max_length=2048,
        batch_size=16,
        device="cpu",
    ):
        LM.__init__(self)

        self._model = MambaDecode.from_pretrained(pretrained)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
