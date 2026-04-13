from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]           # seqlen
    shape[-1] = x.shape[-1]         # head_dim//2
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _, seqlen, _, _ = query.shape
    device = query.device

    # reshape to complex pairs
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # positions
    t = torch.arange(max_seq_len, device=device).float()

    # outer product
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim//2)

    # slice to current sequence
    freqs = freqs[:seqlen]

    # trig
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # broadcast
    cos = reshape_for_broadcast(cos, query_real)
    sin = reshape_for_broadcast(sin, query_real)

    # rotate query
    q_real = query_real * cos - query_imag * sin
    q_imag = query_real * sin + query_imag * cos

    # rotate key
    k_real = key_real * cos - key_imag * sin
    k_imag = key_real * sin + key_imag * cos

    # restore shape
    query_out = torch.stack([q_real, q_imag], dim=-1).flatten(-2)
    key_out = torch.stack([k_real, k_imag], dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)
