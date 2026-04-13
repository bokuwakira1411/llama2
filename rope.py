from typing import Tuple
import torch

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

    # Step 1: compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Step 2: positions
    t = torch.arange(seqlen, device=device).float()

    # Step 3: outer product → angles
    freqs = torch.outer(t, freqs)  # (seqlen, head_dim//2)

    # Step 4: trig functions
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Step 5: reshape for broadcasting
    cos = reshape_for_broadcast(cos, query_real)
    sin = reshape_for_broadcast(sin, query_real)

    # Step 6: apply rotation (query)
    q_real = query_real * cos - query_imag * sin
    q_imag = query_real * sin + query_imag * cos

    # Step 6: apply rotation (key)
    k_real = key_real * cos - key_imag * sin
    k_imag = key_real * sin + key_imag * cos

    # Step 7: restore original shape
    query_out = torch.stack([q_real, q_imag], dim=-1).flatten(-2)
    key_out = torch.stack([k_real, k_imag], dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)
