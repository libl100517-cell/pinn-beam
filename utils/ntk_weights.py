"""Gradient-based adaptive loss weighting.

Two methods:
1. NTK-based (Wang et al., 2021): max_ratio clipped
2. GradNorm: no clip, log-space EMA for stability

For each loss L_i, compute gradient norm: g_i = ||∇_θ L_i||

Adaptive weight: λ_i = g_mean / g_i
This equalizes each loss component's gradient contribution.
"""

from typing import Dict

import torch
import torch.nn as nn
import math


def _compute_grad_norms(
    raw_losses: Dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
) -> Dict[str, float]:
    """Compute ||∇_θ L_i|| for each loss component."""
    norms = {}
    for name, loss in raw_losses.items():
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        norm = math.sqrt(sum(
            g.detach().pow(2).sum().item()
            for g in grads if g is not None
        ))
        norms[name] = norm
    return norms


def compute_ntk_weights(
    raw_losses: Dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
    ema_weights: Dict[str, float] | None = None,
    alpha: float = 0.1,
    max_ratio: float = 100.0,
) -> Dict[str, float]:
    """NTK-based adaptive weights with max_ratio clipping."""
    norms = _compute_grad_norms(raw_losses, parameters)

    norm_values = [n for n in norms.values() if n > 0]
    if not norm_values:
        return {name: 1.0 for name in raw_losses}
    mean_norm = sum(norm_values) / len(norm_values)

    raw_weights = {}
    for name, norm in norms.items():
        if norm > 0:
            w = mean_norm / norm
        else:
            w = max_ratio
        raw_weights[name] = max(1.0 / max_ratio, min(max_ratio, w))

    if ema_weights is None:
        return raw_weights

    new_weights = {}
    for name in raw_weights:
        prev = ema_weights.get(name, raw_weights[name])
        new_weights[name] = (1 - alpha) * prev + alpha * raw_weights[name]
    return new_weights


def compute_gradnorm_weights(
    raw_losses: Dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
    ema_log_weights: Dict[str, float] | None = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """GradNorm-style adaptive weights — no clipping, log-space EMA.

    Uses log-space EMA to handle large dynamic range without clipping.
    weight_i = exp(ema_log(mean_norm / norm_i))

    Parameters
    ----------
    raw_losses : dict
        Unweighted scalar loss tensors.
    parameters : list
        Network parameters.
    ema_log_weights : dict or None
        Previous EMA of log-weights. None on first call.
    alpha : float
        EMA smoothing in log space. Small = more stable.

    Returns
    -------
    new_weights : dict
        Adaptive weights.
    """
    norms = _compute_grad_norms(raw_losses, parameters)

    norm_values = [n for n in norms.values() if n > 0]
    if not norm_values:
        return {name: 1.0 for name in raw_losses}

    # Geometric mean (in log space) for better stability
    log_mean = sum(math.log(n + 1e-30) for n in norm_values) / len(norm_values)

    # log(weight_i) = log(geom_mean) - log(norm_i)
    raw_log_weights = {}
    for name, norm in norms.items():
        if norm > 0:
            raw_log_weights[name] = log_mean - math.log(norm)
        else:
            raw_log_weights[name] = 0.0

    # EMA in log space
    if ema_log_weights is None:
        ema_log_weights = raw_log_weights
    else:
        ema_log_weights = {
            name: (1 - alpha) * ema_log_weights.get(name, raw_log_weights[name])
                  + alpha * raw_log_weights[name]
            for name in raw_log_weights
        }

    # Convert back: weight = exp(log_weight)
    weights = {name: math.exp(lw) for name, lw in ema_log_weights.items()}
    return weights, ema_log_weights
