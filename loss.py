from torch import torch
from torch.nn import functional as F



def mse_token_weight_loss(student_hidden_states, teacher_hidden_states, token_weights):
    squared_diff = (student_hidden_states - teacher_hidden_states) ** 2
    weighted_mse_loss = squared_diff.mean(-1) * token_weights.squeeze()
    weighted_mse_loss = weighted_mse_loss.sum(-1).mean()

    return weighted_mse_loss

def cosine_token_weight_loss(student_hidden_states, teacher_hidden_states, token_weights):
    cos_sim = F.cosine_similarity(student_hidden_states, teacher_hidden_states, dim=-1, eps=1e-5)
    cos_sim_loss = 1 - cos_sim
    weighted_cos_sim_loss = cos_sim_loss * token_weights.squeeze()
    weighted_mse_loss = weighted_cos_sim_loss.sum(-1).mean()

    return weighted_mse_loss

def cosine_loss(student_embeddings, teacher_embeddings):
    cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
    cos_sim_loss = 1 - cos_sim
    return cos_sim_loss.mean()


        