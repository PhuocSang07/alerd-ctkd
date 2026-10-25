import os
import sys

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_span_hidden_states, last_token_rows  # noqa: E402


def tiny_config(kind):
    common = dict(vocab_size=120, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4)
    if kind == "qwen2":
        return transformers.Qwen2Config(hidden_size=32, num_key_value_heads=2, max_position_embeddings=64, **common)
    if kind == "llama":
        return transformers.LlamaConfig(hidden_size=32, num_key_value_heads=2, max_position_embeddings=64, **common)
    if kind == "gpt2":
        return transformers.GPT2Config(vocab_size=120, n_embd=32, n_layer=3, n_head=4, n_positions=64)
    raise ValueError(kind)


def tiny_model(kind):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tiny_config(kind), attn_implementation="eager").eval()


def padded_batch(lengths, vocab=120, total=None):
    total = total or max(lengths)
    ids = torch.randint(1, vocab, (len(lengths), total))
    mask = torch.zeros(len(lengths), total, dtype=torch.long)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        ids[b, n:] = 0
    return ids, mask


def forward(model, ids, mask):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, output_attentions=True, output_hidden_states=True,
                     use_cache=False)


@pytest.mark.parametrize("kind", ["qwen2", "llama", "gpt2"])
def test_last_row_at_the_last_real_token_matches_the_unpadded_sequence(kind):
    model = tiny_model(kind)
    lengths = [9, 5, 12]
    ids, mask = padded_batch(lengths, total=14)
    batched = forward(model, ids, mask).attentions
    for layer in range(3):
        rows = last_token_rows(batched[layer], mask)
        assert rows.shape == (3, 14)
        for b, n in enumerate(lengths):
            alone = forward(model, ids[b:b + 1, :n], torch.ones(1, n, dtype=torch.long)).attentions[layer]
            expected = alone[0, :, n - 1, :].sum(0)  # all heads of the last real token over its n keys
            torch.testing.assert_close(rows[b, :n], expected, atol=1e-5, rtol=1e-4)
            assert (rows[b, n:] == 0).all()  # pad keys carry no mass
            assert rows[b].sum() == pytest.approx(4.0, abs=1e-4)  # 4 heads, each a distribution


@pytest.mark.parametrize("kind", ["qwen2", "llama", "gpt2"])
def test_the_last_column_would_be_the_wrong_row_for_shorter_sequences(kind):
    model = tiny_model(kind)
    ids, mask = padded_batch([5, 12])
    attention = forward(model, ids, mask).attentions[0]
    right = last_token_rows(attention, mask)
    wrong = attention[:, :, -1, :].sum(1)  # [:, -1] is a pad position for the 5-token sequence
    assert not torch.allclose(right[0], wrong[0])
    torch.testing.assert_close(right[1], wrong[1])  # the longest sequence has no padding


@pytest.mark.parametrize("kind", ["qwen2", "llama", "gpt2"])
def test_span_states_and_masses_do_not_depend_on_batching(kind):
    """get_span_hidden_states on a padded batch equals the same call on each sequence alone."""
    model = tiny_model(kind)
    lengths = [10, 6]
    ids, mask = padded_batch(lengths, total=12)
    spans = {0: [(2, 5), (5, 7), (7, 10)], 1: [(1, 3), (3, 6)]}

    def pooler(sample_spans, n_spans, width):
        safe = torch.zeros(1, n_spans, width, dtype=torch.long)
        valid = torch.zeros(1, n_spans, width, dtype=torch.bool)
        for s, (a, b) in enumerate(sample_spans):
            safe[0, s, :b - a] = torch.arange(a, b)
            valid[0, s, :b - a] = True
        return safe, valid

    out = forward(model, ids, mask)
    safe_b = torch.cat([pooler(spans[0], 3, 3)[0], pooler(spans[1], 3, 3)[0]])
    valid_b = torch.cat([pooler(spans[0], 3, 3)[1], pooler(spans[1], 3, 3)[1]])
    pooled, mass = get_span_hidden_states(out.hidden_states, out.attentions, mask, safe_b, valid_b, [2, 3])
    for b, n in enumerate(lengths):
        alone = forward(model, ids[b:b + 1, :n], torch.ones(1, n, dtype=torch.long))
        safe, valid = pooler(spans[b], 3, 3)
        p_alone, m_alone = get_span_hidden_states(alone.hidden_states, alone.attentions,
                                                  torch.ones(1, n, dtype=torch.long), safe, valid, [2, 3])
        n_spans = len(spans[b])
        torch.testing.assert_close(pooled[:, b, :n_spans], p_alone[:, 0, :n_spans], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mass[:, b, :n_spans], m_alone[:, 0, :n_spans], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("kind", ["qwen2", "llama", "gpt2"])
def test_the_last_hidden_state_is_the_normalised_one(kind):
    """HF's last entry already went through the final norm, so lm_head applies to it directly."""
    model = tiny_model(kind)
    ids, mask = padded_batch([7, 7])
    out = forward(model, ids, mask)
    logits = model.get_output_embeddings()(out.hidden_states[3])
    torch.testing.assert_close(logits, out.logits, atol=1e-5, rtol=1e-4)
