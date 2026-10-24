"""The three span-level KD losses of SRA (cosine Eq. 5, geometry Eq. 6-7, KL Eq. 11) with swappable span weights.

`attention` weights are SRA's (Eq. 8): the teacher's last-token attention mass of each span, raised to p and
normalised. Any other source supplies omega (B, N), summing to 1 over the real spans of each sample: it replaces
the weights of the cosine loss, builds the pair weights of the geometry loss, and scales the KL of span n by
N * omega_n so the total KL weight stays N, as in the uniform case.
"""
import torch
import torch.nn.functional as F

EPS = 1e-5


def make_weights(teacher_mass, valid, p=1.0, omega=None):
    """(layer weights (L, B, N), pair weights (B, N, N), KL weights (B, N)).

    teacher_mass: (L, B, N) attention mass of each span; valid: (B, N) bool for real (non-padding) spans.
    """
    n_layers = teacher_mass.size(0)
    validf = valid.float()
    if omega is None:
        layer_w = teacher_mass ** p
        layer_w = layer_w / layer_w.sum(-1, keepdim=True)
        last = layer_w[-1]
        kl_w = validf
    else:
        omega = omega.to(teacher_mass) * validf
        layer_w = omega.unsqueeze(0).expand(n_layers, -1, -1)
        last = omega
        kl_w = validf.sum(-1, keepdim=True) * omega
    pair = last.unsqueeze(2) * last.unsqueeze(1)
    pair = pair * (1 - torch.eye(last.size(1), device=pair.device)).unsqueeze(0)
    pair = pair / pair.sum(dim=(1, 2), keepdim=True).clamp(min=EPS)
    return layer_w, pair, kl_w


def cosine_loss(student_hidden, teacher_hidden, layer_w, layer_loss_weights):
    """sum_l lambda_l * mean_b sum_n w_ln (1 - cos(student, teacher)); hidden: (L, B, N, D)."""
    total = 0
    for i, weight in enumerate(layer_loss_weights):
        cos = F.cosine_similarity(student_hidden[i], teacher_hidden[i], dim=-1, eps=EPS)
        total = total + weight * ((1 - cos) * layer_w[i]).sum(-1).mean()
    return total


def geometry_loss(student_embed, teacher_last, pair):
    """Squared difference of the span-to-span cosine matrices, weighted by pair weights, summed per sample."""
    s = F.normalize(student_embed, dim=-1, eps=EPS)
    t = F.normalize(teacher_last, dim=-1, eps=EPS)
    scores = F.mse_loss(torch.matmul(s, s.transpose(-1, -2)), torch.matmul(t, t.transpose(-1, -2)), reduction="none")
    return (scores * pair).sum() / pair.size(0)


def kl_loss(student_logits, teacher_logits, kl_w, temperature):
    """Weighted KL(teacher || student) per span over the shared vocabulary; summed over spans, mean over batch."""
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    q = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(log_p, q, reduction="none").sum(-1)
    return (kl * kl_w).sum() / student_logits.size(0)


def shared_vocab_logits(hidden, lm_head_weight, lm_head_bias, ids):
    """Logits of the shared-vocabulary rows only: hidden @ W[ids]^T (+ b[ids])."""
    bias = None if lm_head_bias is None else lm_head_bias[ids]
    return F.linear(hidden, lm_head_weight[ids], bias)
