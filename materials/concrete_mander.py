"""Mander constitutive model for unconfined concrete.

Reference
---------
Mander, J. B., Priestley, M. J. N., & Park, R. (1988).
Theoretical stress-strain model for confined concrete.
Journal of Structural Engineering, 114(8), 1804-1826.

Tension softening uses a fracture-energy-based linear descending branch:
    - Linear ascending:  σ = Ec·ε   for 0 < ε ≤ ε_cr  (ε_cr = ft / Ec)
    - Linear softening:  σ = ft·(1 - (ε - ε_cr)/(ε_tu - ε_cr))  for ε_cr < ε ≤ ε_tu
    - Zero:              σ = 0       for ε > ε_tu
    where ε_tu = 2·Gf / (ft·h)  is the ultimate tensile strain from fracture energy.

Sign convention
---------------
Compressive stress and strain are **negative**.
``fc`` is stored as a positive number and internally negated.
"""

from typing import Dict

import torch

from .base_material import BaseMaterial


class ManderConcrete(BaseMaterial):
    """Mander unconfined concrete model with fracture-energy tension softening.

    Parameters
    ----------
    fc : float
        Compressive strength (positive, MPa).
    Ec : float
        Elastic modulus (MPa).
    eps_co : float
        Strain at peak compressive stress (negative, default -0.002).
    eps_cu : float
        Ultimate compressive strain (negative, default -0.004).
    ft : float or None
        Tensile strength (positive, MPa). Default 0.62*sqrt(fc).
    Gf : float
        Fracture energy (N/mm). Default 0.1 N/mm (typical for normal concrete).
    h : float
        Characteristic length / crack band width (mm). Default 150 mm.
    """

    def __init__(
        self,
        fc: float = 30.0,
        Ec: float = 25000.0,
        eps_co: float = -0.002,
        eps_cu: float = -0.004,
        ft: float | None = None,
        Gf: float = 0.1,
        h: float = 150.0,
    ):
        self.fc = fc
        self.Ec = Ec
        self.eps_co = eps_co
        self.eps_cu = eps_cu
        self.ft = ft if ft is not None else 0.62 * fc ** 0.5
        self.Gf = Gf
        self.h = h

    @property
    def eps_cr(self) -> float:
        """Cracking strain = ft / Ec."""
        return self.ft / self.Ec

    @property
    def eps_tu(self) -> float:
        """Ultimate tensile strain from fracture energy: ε_tu = 2·Gf / (ft·h)."""
        return 2.0 * self.Gf / (self.ft * self.h)

    # ------------------------------------------------------------------
    # Mander compression helpers
    # ------------------------------------------------------------------

    def _esec(self) -> float:
        return self.fc / abs(self.eps_co)

    def _r(self) -> float:
        Esec = self._esec()
        return self.Ec / (self.Ec - Esec)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Evaluate stress.

        Compression (ε < 0):
            Mander curve for eps_cu ≤ ε < 0, zero for ε < eps_cu.
        Tension (ε > 0):
            Linear ascending to ft at ε_cr, then linear softening to 0 at ε_tu.
        """
        sigma = torch.zeros_like(strain)
        r = self._r()
        eps_cr = self.eps_cr
        eps_tu = self.eps_tu

        # --- compression: Mander curve ---
        comp_mask = strain < 0.0
        crush_mask = strain < self.eps_cu
        active_comp = comp_mask & ~crush_mask

        if active_comp.any():
            eps_c = strain[active_comp]
            x = eps_c / self.eps_co
            xr = torch.pow(torch.clamp(x, min=1e-12), r)
            sigma[active_comp] = -self.fc * r * x / (r - 1.0 + xr)

        # --- tension: linear ascending ---
        asc_mask = (strain > 0.0) & (strain <= eps_cr)
        if asc_mask.any():
            sigma[asc_mask] = self.Ec * strain[asc_mask]

        # --- tension: fracture energy softening ---
        soft_mask = (strain > eps_cr) & (strain <= eps_tu)
        if soft_mask.any():
            eps_s = strain[soft_mask]
            sigma[soft_mask] = self.ft * (1.0 - (eps_s - eps_cr) / (eps_tu - eps_cr))

        # strain > eps_tu → sigma stays 0 (already initialised)

        return sigma

    def tangent(self, strain: torch.Tensor) -> torch.Tensor:
        """Tangent modulus dσ/dε via automatic differentiation."""
        eps = strain.detach().requires_grad_(True)
        sig = self.stress(eps)
        grad = torch.autograd.grad(
            sig, eps, grad_outputs=torch.ones_like(sig), create_graph=True
        )[0]
        return grad

    def get_parameters(self) -> Dict[str, float]:
        return {
            "fc": self.fc,
            "Ec": self.Ec,
            "eps_co": self.eps_co,
            "eps_cu": self.eps_cu,
            "ft": self.ft,
            "Gf": self.Gf,
            "h": self.h,
        }

    def set_parameters(self, **kwargs: float) -> None:
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, val)
