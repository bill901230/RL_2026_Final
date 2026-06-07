import torch


def rearrange(*args, **kwargs):
    try:
        from einops import rearrange as _rearrange
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("einops is required by the W3.T1 flash_attn shim") from exc
    return _rearrange(*args, **kwargs)


def index_first_axis(x, indices):
    return x[indices]


def unpad_input(hidden_states, attention_mask):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    unpadded = flat[indices]
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return unpadded, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])
