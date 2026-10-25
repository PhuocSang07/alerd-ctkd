import torch


def last_token_rows(attention, attention_mask):
    """(B, T) attention of each sequence's last real token over all keys, summed over heads.

    attention: (B, H, T, T) from `output_attentions`. The last real token is found from the attention mask: with
    right padding `[:, -1]` would be a pad token for every sequence shorter than the batch.
    """
    last = attention_mask.sum(dim=1) - 1
    rows = torch.arange(attention.size(0), device=attention.device)
    return attention[rows, :, last.to(attention.device), :].sum(dim=1)


def get_span_hidden_states(hidden_states, attentions, attention_mask, safe_idx, pooler_mask, hidden_layer_fineturn,
                           weight_pooling=True, span_weight=True):
    """Pool the token states of each span (SRA Eq. 3-4) and measure each span's attention mass (Eq. 8).

    hidden_states: HF `hidden_states` (entry l is the output of layer l, 1-indexed; the last one is normalised);
    attentions: HF `attentions` (entry l - 1 is layer l). Returns (L, B, N, D) pooled states and (L, B, N, 1) masses.
    """
    batch_size = safe_idx.size(0)
    batch_idxs = torch.arange(batch_size, device=safe_idx.device)[:, None, None]

    hidden_state_pools = []
    span_weights = []
    for i in hidden_layer_fineturn:
        weights = last_token_rows(attentions[i - 1], attention_mask).detach()
        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.unsqueeze(-1)[batch_idxs, safe_idx] * pooler_mask.unsqueeze(-1)

        gathered = hidden_states[i][batch_idxs, safe_idx] * pooler_mask.unsqueeze(-1)
        if weight_pooling:
            gathered = gathered * weights
            hidden_state_mean = gathered.sum(2) / weights.sum(2).clamp(min=1e-5)
        else:
            hidden_state_mean = gathered.sum(2) / pooler_mask.sum(2, keepdim=True).clamp(min=1e-5)

        hidden_state_pools.append(hidden_state_mean)

        if span_weight:
            span_weights.append(weights.sum(2))
        else:
            span_weights.append(weights.sum(2) ** 1e-5)

    return torch.stack(hidden_state_pools), torch.stack(span_weights)
