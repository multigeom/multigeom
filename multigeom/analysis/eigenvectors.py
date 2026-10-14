"""Comparing the eigenvectors of two metrics one by one."""
from __future__ import annotations

import torch


def match_eigenvectors(ref: torch.Tensor, tgt: torch.Tensor, n: int, pool: int) -> torch.Tensor:
    """
    For each of the first `n` reference eigenvectors (rows of `ref`), pick the not-yet-used eigenvector among the first
    `pool` rows of `tgt` with the largest abs(cosine), and flip its sign to make the cosine positive. Returns [n, P]. With
    pool = n this only reorders and sign-flips the leading n target eigenvectors. The sign of an eigenvector is arbitrary,
    so this is how two sets are put side by side for display.
    """
    ref = ref.double()
    tgt = tgt.double()
    pool = max(min(pool, tgt.shape[0]), min(n, tgt.shape[0]))
    used: set[int] = set()
    out = []
    for i in range(min(n, ref.shape[0], tgt.shape[0])):
        cos = tgt[:pool] @ ref[i]
        score = cos.abs().clone()
        for u in used:
            score[u] = -1.0
        j = int(score.argmax())
        used.add(j)
        out.append(tgt[j] if cos[j] >= 0 else -tgt[j])
    return torch.stack(out).float()
