"""Hooks that keep only what the span losses need from a decoder-only LM.

Every layer runs fused SDPA attention. For the distilled layers, a forward hook stores the layer output and
recomputes the attention row of the last real token (softmax(q_N K^T / sqrt(d)) with RoPE and GQA, summed over
heads, detached): SRA Eq. 8 only reads that row, so the (T x T) attention matrix is never materialised.

Layers are 1-indexed like `teacher_layers_mapping`. The output of the last layer goes through the final norm,
as in HF `output_hidden_states`, so `lm_head` applies to it directly. Right padding is assumed.
"""
import torch
from torch import nn


def find_decoder_layers(model):
    """(layers, final_norm) of a Llama-style decoder, also through a PEFT wrapper."""
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, nn.ModuleList) and len(module) > 0 \
                and hasattr(module[0], "self_attn"):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            return module, getattr(parent, "norm", None)
    raise NotImplementedError("no decoder layers with `self_attn` found; only Llama-style models are supported")


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(attn, hidden, kwargs):
    """cos, sin of shape (B or 1, T, head_dim): the model's own tables when passed, else default RoPE."""
    embeddings = kwargs.get("position_embeddings")
    if embeddings is not None:
        return embeddings
    config = attn.config
    if getattr(config, "rope_scaling", None):
        raise NotImplementedError("rope_scaling is only supported through `position_embeddings`")
    head_dim = getattr(attn, "head_dim", config.hidden_size // config.num_attention_heads)
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device=hidden.device).float() / head_dim))
    freqs = position_ids[..., None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def last_row_attention(attn, hidden, kwargs, last_index):
    """Head-summed attention row (B, T) of token `last_index[b]` of each sequence, from the attention module's input."""
    config = attn.config
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(attn, "head_dim", config.hidden_size // n_heads)
    batch, length, _ = hidden.shape
    rows = torch.arange(batch, device=hidden.device)

    q = attn.q_proj(hidden[rows, last_index]).view(batch, 1, n_heads, head_dim)
    k = attn.k_proj(hidden).view(batch, length, n_kv, head_dim)
    if hasattr(attn, "q_norm"):  # Qwen3 normalises q and k per head before RoPE
        q, k = attn.q_norm(q), attn.k_norm(k)

    cos, sin = rope_tables(attn, hidden, kwargs)
    cos, sin = cos.expand(batch, -1, -1).to(q.dtype), sin.expand(batch, -1, -1).to(q.dtype)
    q = q * cos[rows, last_index][:, None, None] + rotate_half(q) * sin[rows, last_index][:, None, None]
    k = k * cos[:, :, None] + rotate_half(k) * sin[:, :, None]

    k = k.repeat_interleave(n_heads // n_kv, dim=2)
    scaling = getattr(attn, "scaling", head_dim ** -0.5)
    scores = torch.einsum("bqhd,bthd->bht", q, k).float() * scaling
    keys = torch.arange(length, device=hidden.device)[None]
    scores = scores.masked_fill(keys[:, None, :] > last_index[:, None, None], float("-inf"))
    return scores.softmax(-1).sum(1)


class LayerCapture:
    """Stores `hidden[l]` (layer output) and `rows[l]` (last-token attention row) for the chosen layers."""

    def __init__(self, model, layers, attention_rows=True):
        self.layers = list(layers)
        decoder_layers, self.final_norm = find_decoder_layers(model)
        self.n_layers = len(decoder_layers)
        if any(not 1 <= l <= self.n_layers for l in self.layers):
            raise ValueError(f"layers {self.layers} out of range for a {self.n_layers}-layer model")
        self.hidden, self.rows = {}, {}
        self.last_index = None
        self.enabled = False
        for l in self.layers:
            layer = decoder_layers[l - 1]
            layer.register_forward_hook(self._store_hidden(l))
            if attention_rows:
                layer.self_attn.register_forward_pre_hook(self._store_row(l), with_kwargs=True)

    def begin(self, attention_mask):
        self.hidden.clear()
        self.rows.clear()
        self.last_index = attention_mask.sum(dim=1) - 1
        self.enabled = True

    def end(self):
        self.enabled = False

    def _store_hidden(self, l):
        def hook(module, args, output):
            if self.enabled:  # re-running a checkpointed layer during backward overwrites with the same value
                self.hidden[l] = output[0] if isinstance(output, tuple) else output
        return hook

    def _store_row(self, l):
        def hook(module, args, kwargs):
            if not self.enabled or l in self.rows:
                return
            hidden = args[0] if args else kwargs["hidden_states"]
            # A detached statistic: dropout off, autograd off and autocast off. Running the projections under
            # autocast would cache detached bf16 copies of the weights that the real forward then reuses
            # (no gradient to the weights), and drawing random numbers would desynchronise a checkpointed
            # layer's recomputation.
            modes = [(m, m.training) for m in (module.q_proj, module.k_proj)]
            for m, _ in modes:
                m.eval()
            hidden = hidden.detach().to(module.q_proj.weight.dtype)
            with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
                self.rows[l] = last_row_attention(module, hidden, kwargs, self.last_index)
            for m, mode in modes:
                m.train(mode)
        return hook

    def hidden_states(self):
        """{layer: (B, T, D)}; the final layer is normalised."""
        out = dict(self.hidden)
        if self.n_layers in out and self.final_norm is not None:
            out[self.n_layers] = self.final_norm(out[self.n_layers])
        return out
