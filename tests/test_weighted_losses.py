import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kd_losses import (cosine_loss, geometry_loss, kl_loss, make_weights,  # noqa: E402
                       shared_vocab_logits)

L, B, N, DS, DT, V = 2, 3, 6, 5, 7, 11
LAYER_LOSS_WEIGHTS = [0.4, 0.6]


def tensors(seed=0, n_valid=(6, 4, 5)):
    g = torch.Generator().manual_seed(seed)
    valid = torch.zeros(B, N, dtype=torch.bool)
    for b, n in enumerate(n_valid):
        valid[b, :n] = True
    mass = torch.rand(L, B, N, 1, generator=g) * valid.unsqueeze(0).unsqueeze(-1)
    student_hidden = torch.randn(L, B, N, DT, generator=g) * valid[None, ..., None]
    teacher_hidden = torch.randn(L, B, N, DT, generator=g) * valid[None, ..., None]
    student_embed = torch.randn(B, N, DS, generator=g) * valid[..., None]
    s_logits = torch.randn(B, N, V, generator=g)
    t_logits = torch.randn(B, N, V, generator=g)
    return valid, mass, student_hidden, teacher_hidden, student_embed, s_logits, t_logits


def original_sra_losses(span_mass, student_hidden, teacher_hidden, student_embed, teacher_last, s_logits, t_logits,
                        p, temperature):
    """The loss code as it was before span weights became swappable (llm_train.knowledge_distillation_loss)."""
    span_weights = span_mass.squeeze(-1)
    _, batch, n = span_weights.size()
    span_weights = span_weights ** p
    span_weights = span_weights / span_weights.sum(-1, keepdim=True)
    pair_weights = span_weights[-1].unsqueeze(2) * span_weights[-1].unsqueeze(1)
    mask = torch.eye(n).bool()
    pair_weights[:, mask] = 0.0
    pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    span_weights = span_weights.unsqueeze(-1)

    span_loss = 0
    for i in range(span_weights.size(0)):
        cos = F.cosine_similarity(student_hidden[i], teacher_hidden[i], dim=-1, eps=1e-5)
        weighted = (1 - cos) * span_weights[i].squeeze()
        span_loss += LAYER_LOSS_WEIGHTS[i] * weighted.sum(-1).mean()

    s_hidden = F.normalize(student_embed, dim=-1, eps=1e-5)
    t_hidden = F.normalize(teacher_last, dim=-1, eps=1e-5)
    score_loss = F.mse_loss(torch.matmul(s_hidden, s_hidden.transpose(-1, -2)),
                            torch.matmul(t_hidden, t_hidden.transpose(-1, -2)), reduction="none")
    geometry = (score_loss * pair_weights).sum() / batch

    log_p = F.log_softmax(s_logits / temperature, dim=-1)
    q = F.softmax(t_logits / temperature, dim=-1)
    mask = (s_logits.abs().sum(dim=-1) != 0).float()
    kl = (F.kl_div(log_p, q, reduction="none").sum(dim=-1) * mask).sum() / s_logits.size(0)
    return span_loss, geometry, kl


def new_losses(mass, valid, omega, student_hidden, teacher_hidden, student_embed, s_logits, t_logits, p=1.0, T=2.0):
    layer_w, pair_w, kl_w = make_weights(mass.squeeze(-1), valid, p, omega)
    return (cosine_loss(student_hidden, teacher_hidden, layer_w, LAYER_LOSS_WEIGHTS),
            geometry_loss(student_embed, teacher_hidden[-1], pair_w),
            kl_loss(s_logits, t_logits, kl_w, T))


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_attention_source_equals_the_original_losses(p):
    valid, mass, sh, th, se, sl, tl = tensors()
    sl, tl = sl * valid[..., None], tl * valid[..., None]
    expected = original_sra_losses(mass, sh, th, se, th[-1], sl, tl, p, 2.0)
    got = new_losses(mass, valid, None, sh, th, se, sl, tl, p)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b)


def test_uniform_omega_equals_attention_source_with_uniform_mass():
    valid, _, sh, th, se, sl, tl = tensors()
    sl, tl = sl * valid[..., None], tl * valid[..., None]
    uniform_mass = valid.float().unsqueeze(0).unsqueeze(-1).expand(L, B, N, 1)
    omega = valid.float() / valid.sum(-1, keepdim=True)
    via_attention = new_losses(uniform_mass, valid, None, sh, th, se, sl, tl)
    via_omega = new_losses(uniform_mass, valid, omega, sh, th, se, sl, tl)
    for a, b in zip(via_omega, via_attention):
        torch.testing.assert_close(a, b)


def test_kl_weight_total_is_the_number_of_spans():
    valid, mass, *_ = tensors()
    omega = torch.rand(B, N) * valid
    omega = omega / omega.sum(-1, keepdim=True)
    _, _, kl_w = make_weights(mass.squeeze(-1), valid, 1.0, omega)
    torch.testing.assert_close(kl_w.sum(-1), valid.sum(-1).float())


def test_shared_vocab_rows_equal_slicing_full_logits():
    g = torch.Generator().manual_seed(1)
    hidden, weight, bias = torch.randn(2, 4, DS, generator=g), torch.randn(30, DS, generator=g), torch.randn(30, generator=g)
    ids = torch.tensor([3, 7, 8, 21, 29])
    full = F.linear(hidden, weight, bias)[:, :, ids]
    torch.testing.assert_close(shared_vocab_logits(hidden, weight, bias, ids), full)
    torch.testing.assert_close(shared_vocab_logits(hidden, weight, None, ids), F.linear(hidden, weight)[:, :, ids])
