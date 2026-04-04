"""Smooth nonlinear concrete — differentiable via torch.where.

Compression (ε ≤ 0): Popovics curve
    σ = -fc · n · η / (n - 1 + η^n)
    η = -ε / |ε_co|,  n = Ec · |ε_co| / fc
    Exact: σ(ε_co) = -fc, dσ/dε(0⁻) = Ec

Tension (ε > 0): Linear ascending + sharp exponential softening
    σ = Ec · ε · exp(-relu_smooth(ε - ε_cr) / ε_f)
    relu_smooth(x) = softplus(x·β)/β  with large β
    Exact: dσ/dε(0⁺) = Ec, peak ≈ ft at ε = ε_cr

torch.where supports autograd — gradients flow correctly.
At ε = 0 both branches give 0, so the value is continuous.
The tangent has a tiny discontinuity at ε = 0 only when n ≠ Ec·|ε_co|/fc,
which is never the case here.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .base_material import BaseMaterial


class SmoothConcrete(BaseMaterial):
    """Smooth concrete: Popovics compression + exponential tension.

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
    eps_f : float or None
        Tension softening decay scale. Default ε_cr.
    """

    def __init__(
        self,
        fc: float = 30.0,
        Ec: float = 30000.0,
        eps_co: float = -0.002,
        ft: float | None = None,
        eps_f: float | None = None,
    ):
        self.fc = fc
        self.Ec = Ec
        self.eps_co = eps_co
        self.ft = ft if ft is not None else 0.62 * fc ** 0.5
        self.eps_f = eps_f if eps_f is not None else self.ft / self.Ec

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

        # --- Tension: Ec·ε · exp(-sharp_relu(ε - ε_cr) / ε_f) ---
        # Sharp softplus so decay ≈ 0 for ε < ε_cr, ≈ (ε-ε_cr) for ε > ε_cr
        beta = 100.0 / eps_cr  # sharpness: transition over ~1% of ε_cr
        relu_smooth = F.softplus((strain - eps_cr) * beta) / beta
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
