"""Model loading, last-token hidden capture at all layers, and steered generation.

Layer convention: hidden index l in 0..L, where 0 = embedding output and
l>=1 = output of decoder layer l-1. Injection "at layer l" hooks decoder
layer l-1's output, so it matches activations captured at hidden index l.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_PATHS


def _batched(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


class HookedLM:
    def __init__(self, model_key: str, dtype=torch.bfloat16, device="cuda"):
        path = str(MODEL_PATHS[model_key])
        self.key = model_key
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self.model.eval()
        self.layers = self.model.model.layers
        self.L = len(self.layers)
        self.d = self.model.config.hidden_size

    @torch.no_grad()
    def last_token_hiddens(self, prompts, batch_size=8):
        """Returns [N, L+1, d] float32 CPU tensor of last-token hidden states."""
        outs = []
        for chunk in _batched(prompts, batch_size):
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
            outs.append(hs.float().cpu())
            del out
        return torch.cat(outs)

    def _steer_hook(self, op):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h[:, -1, :] = op(h[:, -1, :]).to(h.dtype)
        return hook

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=8, batch_size=16, op=None, layer=None):
        """Greedy generation; if op is given, it is applied to the last-token
        hidden state at `layer` (1..L) on every forward (prefill + each step)."""
        handle = None
        if op is not None:
            assert 1 <= layer <= self.L
            handle = self.layers[layer - 1].register_forward_hook(self._steer_hook(op))
        try:
            texts = []
            for chunk in _batched(prompts, batch_size):
                enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
                gen = out[:, enc.input_ids.shape[1]:]
                texts.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        finally:
            if handle is not None:
                handle.remove()
        return texts
