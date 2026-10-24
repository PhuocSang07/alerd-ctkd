import os
import sys

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer_capture import LayerCapture  # noqa: E402


def tiny_config(kind):
    common = dict(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    if kind == "qwen2":
        return transformers.Qwen2Config(**common)
    if kind == "llama":
        return transformers.LlamaConfig(**common)
    if kind == "qwen3":
        if not hasattr(transformers, "Qwen3Config"):
            pytest.skip("Qwen3 needs a newer transformers")
        return transformers.Qwen3Config(head_dim=8, **common)
    raise ValueError(kind)


def make_models(kind):
    config = tiny_config(kind)
    torch.manual_seed(0)
    fused = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    eager = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    eager.load_state_dict(fused.state_dict())
    return fused, eager


def padded_batch(lengths, vocab=120, total=None):
    total = total or max(lengths)
    ids = torch.randint(1, vocab, (len(lengths), total))
    mask = torch.zeros(len(lengths), total, dtype=torch.long)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        ids[b, n:] = 0
    return ids, mask


@pytest.mark.parametrize("kind", ["qwen2", "llama", "qwen3"])
def test_hook_row_matches_eager_attention_at_last_real_token(kind):
    fused, eager = make_models(kind)
    ids, mask = padded_batch([9, 5, 12], total=14)
    capture = LayerCapture(fused, [1, 3])
    capture.begin(mask)
    with torch.no_grad():
        fused.model(input_ids=ids, attention_mask=mask, use_cache=False)
    capture.end()
    with torch.no_grad():
        reference = eager(input_ids=ids, attention_mask=mask, output_attentions=True).attentions
    last = mask.sum(1) - 1
    for layer in (1, 3):
        for b in range(len(last)):
            expected = reference[layer - 1][b, :, last[b], :].sum(0)
            torch.testing.assert_close(capture.rows[layer][b, :], expected, atol=1e-5, rtol=1e-4)
        assert (capture.rows[layer][0, int(last[0]) + 1:] == 0).all()  # pad keys carry no mass


def test_hidden_matches_hf_hidden_states_including_final_norm():
    fused, _ = make_models("qwen2")
    ids, mask = padded_batch([7, 10])
    capture = LayerCapture(fused, [2, 3])
    capture.begin(mask)
    with torch.no_grad():
        out = fused.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    capture.end()
    hidden = capture.hidden_states()
    torch.testing.assert_close(hidden[2], out.hidden_states[2])
    torch.testing.assert_close(hidden[3], out.hidden_states[3])  # HF applies the final norm to the last entry


def test_only_requested_layers_are_stored_and_state_resets():
    fused, _ = make_models("llama")
    ids, mask = padded_batch([6, 6])
    capture = LayerCapture(fused, [2])
    for _ in range(2):
        capture.begin(mask)
        with torch.no_grad():
            fused.model(input_ids=ids, attention_mask=mask, use_cache=False)
        capture.end()
        assert list(capture.hidden) == [2] and list(capture.rows) == [2]
    capture.begin(mask)
    capture.end()
    with torch.no_grad():
        fused.model(input_ids=ids, attention_mask=mask, use_cache=False)
    assert capture.hidden == {}  # disabled outside begin/end


def test_gradient_checkpointing_keeps_a_single_value_per_layer():
    fused, _ = make_models("qwen2")
    fused.train()
    fused.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids, mask = padded_batch([8, 8])
    capture = LayerCapture(fused, [2])
    capture.begin(mask)
    out = fused.model(input_ids=ids, attention_mask=mask, use_cache=False)
    capture.end()
    capture.hidden_states()[2].sum().backward()
    assert capture.hidden[2].grad_fn is not None
    assert out.last_hidden_state.requires_grad


def test_base_weights_receive_gradients_under_autocast_with_the_hooks_attached():
    """The row is computed without autocast: under autocast it would cache detached weight copies."""
    fused, _ = make_models("qwen2")
    fused.train()
    fused.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids, mask = padded_batch([8, 6])
    capture = LayerCapture(fused, [2, 3])
    capture.begin(mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = fused(input_ids=ids, attention_mask=mask, use_cache=False).logits
    capture.end()
    logits.float().sum().backward()
    for layer in (1, 2, 3):
        grad = fused.model.layers[layer - 1].self_attn.q_proj.weight.grad
        assert grad is not None and grad.abs().sum() > 0
