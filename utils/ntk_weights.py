"""NTK-based adaptive loss weighting (Wang et al., 2021).

Computes the trace of the Neural Tangent Kernel for each loss component
and uses it to balance learning rates across loss terms.

For each unweighted loss L_i, the NTK trace proxy is:
    K_ii = ||∇_θ L_i||²

Adaptive weight for component i:
    λ_i = mean(K) / K_ii

This ensures all loss components have comparable effective learning rates.
An exponential moving average smooths the weights across updates.
Weights are clipped to [1/max_ratio, max_ratio] for stability.
"""

from typing import Dict

import torch
import torch.nn as nn


def compute_ntk_weights(
    raw_losses: Dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
    ema_weights: Dict[str, float] | None = None,
    alpha: float = 0.1,
    max_ratio: float = 100.0,
) -> Dict[str, float]:
    """Compute NTK-based adaptive weights.

    Parameters
    ----------
    raw_losses : dict
        Unweighted scalar loss tensors (with grad graphs attached).
    parameters : list
        Network parameters to differentiate against.
    ema_weights : dict or None
        Previous EMA weights. None on first call.
    alpha : float
        EMA smoothing factor. Higher = faster adaptation.
    max_ratio : float
        Maximum allowed weight ratio for stability.

    Returns
    -------
    new_weights : dict
        Updated adaptive weights for each loss component.
    """
    # Compute ||∇_θ L_i||² for each component
    traces = {}
    for name, loss in raw_losses.items():
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        trace = sum(
            g.detach().pow(2).sum().item()
            for g in grads if g is not None
        )
        traces[name] = trace

    # Use mean trace as reference (more stable than max)
    trace_values = [t for t in traces.values() if t > 0]
    if not trace_values:
        return {name: 1.0 for name in raw_losses}
    mean_trace = sum(trace_values) / len(trace_values)

    # Adaptive weights: λ_i = mean(K) / K_i, clipped
    raw_weights = {}
    for name, trace in traces.items():
        if trace > 0:
            w = mean_trace / trace
        else:
            w = max_ratio
        raw_weights[name] = max(1.0 / max_ratio, min(max_ratio, w))

    # EMA smoothing
    if ema_weights is None:
        return raw_weights

    new_weights = {}
    for name in raw_weights:
        prev = ema_weights.get(name, raw_weights[name])
        new_weights[name] = (1 - alpha) * prev + alpha * raw_weights[name]
    return new_weights
