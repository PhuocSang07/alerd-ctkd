import torch


def get_span_hidden_states(hidden_states, attention_rows, safe_idx, pooler_mask, layers,
                           weight_pooling=True, span_weight=True):
    """Pool the token states of each span (SRA Eq. 3-4) and measure each span's attention mass (Eq. 8).

    hidden_states: {layer: (B, T, D)}; attention_rows: {layer: (B, T)} head-summed last-token attention rows
    (layer_capture.LayerCapture). Returns (L, B, N, D) pooled states and (L, B, N, 1) span masses.
    """
    batch_size = safe_idx.size(0)
    batch_idxs = torch.arange(batch_size, device=safe_idx.device)[:, None, None]

    hidden_state_pools = []
    span_weights = []
    for i in layers:
        weights = attention_rows[i].detach()
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
