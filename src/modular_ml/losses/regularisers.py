"""General-purpose regulariser functions used as loss criteria.

All operate on a single tensor and return a scalar. They are domain-agnostic
(no motor-unit / recruitment specifics): smoothness, monotonicity, trend,
row-norm uniformity, and effective rank (participation ratio).
"""

import torch


def _temporal_smoothness_loss(x: torch.Tensor, dim: int = 0, p: float = 2.0) -> torch.Tensor:
    """Penalise abrupt changes along ``dim`` (default: time)."""
    if x.shape[dim] < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    dx = torch.diff(x, dim=dim)
    return dx.abs().mean() if p == 1.0 else dx.pow(2).mean()


def _monotonicity_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalise decreases along the last dim (encourage monotone increase)."""
    diff = x[..., 1:] - x[..., :-1]
    return torch.relu(-diff).mean()


def _trend_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalise a positive linear trend along the last dim."""
    n = x.shape[-1]
    idx = torch.arange(n, dtype=x.dtype, device=x.device)
    idx = (idx - idx.mean()) / (idx.std() + 1e-8)
    return torch.relu((x * idx).mean())


def _norm_per_row(x: torch.Tensor) -> torch.Tensor:
    """Encourage uniform row L2 norms (relative-error penalty)."""
    row_norms = torch.linalg.vector_norm(x, ord=2, dim=1)
    tgt = row_norms.mean().detach()
    rel_err = (row_norms - tgt) / (tgt.abs() + 1e-6)
    return rel_err.pow(2).sum()


def _effective_rank(x: torch.Tensor) -> torch.Tensor:
    """Participation ratio of the singular values of a 2D (or flattened) tensor.

    PR = (sum s_i^2)^2 / sum s_i^4, a smooth proxy for the number of effective
    dimensions. Returned negated so that *minimising* the loss *increases* the
    effective rank (a common regularisation target); flip ``reg_weight`` sign in
    config to compress instead.
    """
    x2 = x.reshape(-1, x.shape[-1])
    x2 = x2 - x2.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x2)
    s2 = s.pow(2)
    pr = s2.sum().pow(2) / (s2.pow(2).sum() + 1e-12)
    return -pr


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that tolerates sequence ([T,B,C]) or tabular ([B,C]) logits."""
    c = logits.shape[-1]
    logits_flat = logits.reshape(-1, c)
    labels_flat = labels.reshape(-1).long()
    return torch.nn.functional.cross_entropy(logits_flat, labels_flat)
