import torch
import math

def build_rope_freqs(head_dim, max_seq_len=None, base=10000.0):
    """
    Build RoPE frequency tensor as used in LLaMA/llama.cpp.
    Returns a 1D tensor of shape (rope_dim,), where rope_dim = head_dim // 2.
    """
    rope_dim = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, rope_dim).float() / rope_dim))
    return theta  # shape: [rope_dim]