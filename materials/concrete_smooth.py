"""Smooth nonlinear concrete — differentiable via torch.where.

Compression (ε ≤ 0): Popovics curve
    σ = -fc · n · η / (n - 1 + η^n)
    η = -ε / |ε_co|,  n = Ec · |ε_co| / fc
    Exact: σ(ε_co) = -fc, dσ/dε(0⁻) = Ec

Tension (ε > 0), two modes:
  (a) Exponential softening (default):
      σ = Ec · ε · exp(-relu_smooth(ε - ε_cr) / ε_f)
  (b) Tension stiffening (Vecchio-Collins inspired):
      σ = Ec · ε / (1 + √(α · relu_smooth(ε - ε_cr)))
      Smooth, never reaches zero, captures bond effect with rebar.
      α = 500 by default (Vecchio & Collins, 1986).
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .base_material import BaseMaterial


class SmoothConcrete(BaseMaterial):
    """Smooth concrete: Popovics compression + configurable tension.

    Parameters
    ----------
    fc : float
        Compressive strength (positive, MPa).
    Ec : float
        Elastic modulus (MPa).
    eps_co : float
        Strain at peak compressive stress (negative, default -0.002).
    ft : float or None
        Tensile strength (positive, MPa). Default 0.62*sqrt(fc).
    tension_model : str
        "exp" for exponential softening, "stiffening" for Vecchio-Collins.
    eps_f : float or None
        Softening decay scale (only for "exp" model). Default ε_cr.
    alpha_ts : float
        Tension stiffening parameter (only for "stiffening" model).
        Default 500 (Vecchio & Collins, 1986).
    """

    def __init__(
        self,
        fc: float = 30.0,
        Ec: float = 30000.0,
        eps_co: float = -0.002,
        ft: float | None = None,
        tension_model: str = "exp",
        eps_f: float | None = None,
        alpha_ts: float = 500.0,
    ):
        self.fc = fc
        self.Ec = Ec
        self.eps_co = eps_co
        self.ft = ft if ft is not None else 0.62 * fc ** 0.5
        self.tension_model = tension_model
        self.eps_f = eps_f if eps_f is not None else self.ft / self.Ec
        self.alpha_ts = alpha_ts

    @property
    def n(self) -> float:
        return self.Ec * abs(self.eps_co) / self.fc

    @property
    def eps_cr(self) -> float:
        return self.ft / self.Ec

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        n = self.n
        eps_co = abs(self.eps_co)
        eps_cr = self.eps_cr

        # --- Compression: Popovics ---
        eta = (-strain / eps_co).clamp(min=1e-30)
        sigma_c = -self.fc * n * eta / (n - 1.0 + torch.pow(eta, n))

        # --- Tension ---
        beta = 100.0 / eps_cr
        relu_smooth = F.softplus((strain - eps_cr) * beta) / beta

        if self.tension_model == "stiffening":
            # Tension stiffening (Vecchio-Collins inspired):
            #   ε ≤ ε_cr: σ = Ec·ε (linear elastic)
            #   ε > ε_cr: σ = ft / (1 + √(α · (ε - ε_cr)))
            # Smooth version: cap numerator at ft using smooth min
            sigma_linear = self.Ec * strain
            # Smooth min(Ec·ε, ft): ft - softplus(ft - Ec·ε) ≈ Ec·ε for small ε, ≈ ft for large ε
            beta_cap = 100.0 / self.ft
            sigma_cap = self.ft - F.softplus((self.ft - sigma_linear) * beta_cap) / beta_cap
            # Denominator: 1 + √(α · max(0, ε - ε_cr))
            sigma_t = sigma_cap / (1.0 + torch.sqrt(self.alpha_ts * relu_smooth + 1e-20))
        else:
            # Exponential softening
            sigma_t = self.Ec * strain * torch.exp(-relu_smooth / self.eps_f)

        return torch.where(strain <= 0, sigma_c, sigma_t)

    def tangent(self, strain: torch.Tensor) -> torch.Tensor:
        eps = strain.detach().requires_grad_(True)
        sig = self.stress(eps)
        return torch.autograd.grad(
            sig, eps, grad_outputs=torch.ones_like(sig), create_graph=True
        )[0]

    def get_parameters(self) -> Dict[str, float]:
        return {"fc": self.fc, "Ec": self.Ec, "eps_co": self.eps_co, "ft": self.ft}

    def set_parameters(self, **kwargs: float) -> None:
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, val)
