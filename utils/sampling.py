"""Collocation point sampling utilities."""

import torch


def uniform_collocation(n_points: int, x_min: float = 0.0, x_max: float = 1.0,
                        requires_grad: bool = True,
                        device: torch.device | None = None) -> torch.Tensor:
    """Generate uniformly spaced collocation points in [x_min, x_max].

    Returns tensor of shape (n_points, 1).
    """
    x = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    x.requires_grad_(requires_grad)
    return x


def boundary_points(x_min: float = 0.0, x_max: float = 1.0,
                    requires_grad: bool = True,
                    device: torch.device | None = None) -> torch.Tensor:
    """Return boundary points [x_min, x_max] as tensor of shape (2, 1)."""
    x = torch.tensor([[x_min], [x_max]], dtype=torch.float32, device=device)
    x.requires_grad_(requires_grad)
    return x


def residual_resample(
    n_points: int,
    pointwise_residual: torch.Tensor,
    xi_old: torch.Tensor,
    x_min: float = 0.0,
    x_max: float = 1.0,
    uniform_ratio: float = 0.5,
) -> torch.Tensor:
    """Residual-based adaptive resampling (RAR).

    Generates new collocation points: a fraction from uniform distribution,
    the rest sampled with probability proportional to pointwise residuals.

    Parameters
    ----------
    n_points : int
        Total number of collocation points.
    pointwise_residual : torch.Tensor
        Per-point residual from current collocation, shape (n_col, 1).
    xi_old : torch.Tensor
        Current collocation points, shape (n_col, 1).
    x_min, x_max : float
        Domain bounds.
    uniform_ratio : float
        Fraction of points sampled uniformly (rest from residual distribution).

    Returns
    -------
    xi_new : torch.Tensor
        New collocation points, shape (n_points, 1), with requires_grad=True.
    """
    device = xi_old.device
    n_uniform = int(n_points * uniform_ratio)
    n_adaptive = n_points - n_uniform

    # Uniform part
    xi_uniform = torch.rand(n_uniform, 1, device=device) * (x_max - x_min) + x_min

    # Adaptive part: sample from residual distribution
    res = pointwise_residual.flatten()
    prob = res / res.sum()
    idx = torch.multinomial(prob, n_adaptive, replacement=True)
    # Add small perturbation to avoid exact duplicates
    xi_adaptive = xi_old[idx].detach().clone()
    noise = (torch.rand_like(xi_adaptive) - 0.5) * 2 * (x_max - x_min) / n_points
    xi_adaptive = (xi_adaptive + noise).clamp(x_min, x_max)

    xi_new = torch.cat([xi_uniform, xi_adaptive], dim=0)
    xi_new = xi_new.sort(dim=0).values
    xi_new.requires_grad_(True)
    return xi_new
