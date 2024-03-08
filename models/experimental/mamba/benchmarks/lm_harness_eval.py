from typing import List, Tuple
from tqdm import tqdm

import torch

from transformers import AutoTokenizer

from models.experimental.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba-cpu-reference")
class MambaEvalWrapper(LM):
    def __init__(
        self,
        pretrained: MambaPretrainedModelName = "state-spaces/mamba-2.8b",
        max_length=2048,
        batch_size=1,
        device="cpu",
    ):
        LM.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self.model = MambaDecode.from_pretrained(pretrained, batch_size=int(batch_size))

        self.max_length = int(max_length)
        self.device = torch.device(device)

    def loglikelihood(self, requests: List[Instance]):
        results = []
        with torch.no_grad():
            for instance in tqdm(requests):
                context, target = instance.arguments

                # Reset the model hidden/conv states before decoding
                self.model.initialize_states()

                context_ids = self.tokenizer(context, return_tensors="pt").input_ids  # (1 x CONTEXT_LEN)
                if context == "":
                    context_ids = torch.Tensor([self.tokenizer.eos_token_id])
                assert len(context_ids.shape) == 2 and context_ids.shape[1] > 0, "Expected at least one context token"

                target_ids = self.tokenizer(target, return_tensors="pt").input_ids  # (1 x TARGET_LEN)
                assert len(target_ids.shape) == 2 and target_ids.shape[1] > 0, "Expected at least one target token"

                # Prefill using decode. Put all tokens in the prompt through the model except the last one
                num_prefill_tokens = context_ids.shape[1] - 1
                for idx in range(num_prefill_tokens):
                    self.model(context_ids[:, idx].unsqueeze(1))  # Model expects (B x 1 x 1)

                # For each target token we should predict using the preceding one
                num_target_tokens = target_ids.shape[1]
                last_token = context_ids[:, -1].unsqueeze(1)  # Model expects (Bx1)
                logits = []
                is_greedy = True
                for idx in range(num_target_tokens):
                    out = self.model(last_token)  # (B x 1) => (B x 1 x VOCAB)

                    probs = torch.nn.functional.log_softmax(out, dim=-1)
                    logits.append(probs)

                    last_token = torch.argmax(out, dim=-1)
                    target_token = target_ids[:, idx].unsqueeze(0)
                    assert (
                        last_token.shape == target_token.shape
                    ), f"Expected actual and target token to be same shape ({last_token.shape} vs. {target_token.shape})"
                    if last_token != target_token:
                        is_greedy = False

                # Compute loglikelihood using the recorded logits
                loglikelihood = 0.0
                for idx in range(num_target_tokens):
                    prob = logits[idx]
                    target_token = target_ids[:, idx]
                    loglikelihood += prob[0, 0, target_token[0]].item()

                results.append((loglikelihood, is_greedy))

        return results

    def generate_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
