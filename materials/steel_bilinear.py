"""Bilinear elasto-plastic constitutive model for reinforcing steel.

The model has:
- a linear elastic branch with modulus Es
- a yield plateau / hardening branch beyond fy

Post-yield behaviour is controlled by the hardening ratio ``b``:
    E_hardening = b * Es

Setting b = 0 gives perfect plasticity.
"""

from typing import Dict

import torch

from .base_material import BaseMaterial


class BilinearSteel(BaseMaterial):
    """Bilinear steel model (tension and compression symmetric).

    Parameters
    ----------
    fy : float
        Yield strength (MPa).
    Es : float
        Elastic modulus (MPa).
    b : float
        Hardening ratio (E_hard / Es).  Default 0.01.
    eps_u : float
        Ultimate strain (positive).  Default 0.1.
    """

    def __init__(
        self,
        fy: float = 400.0,
        Es: float = 200000.0,
        b: float = 0.01,
        eps_u: float = 0.1,
    ):
        self.fy = fy
        self.Es = Es
        self.b = b
        self.eps_u = eps_u

    @property
    def eps_y(self) -> float:
        """Yield strain."""
        return self.fy / self.Es

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Evaluate bilinear steel stress (symmetric in tension/compression)."""
        eps_y = self.eps_y
        E_hard = self.b * self.Es

        # Elastic branch
        sigma_elastic = self.Es * strain

        # Post-yield branch (tension side)
        sigma_hard_pos = self.fy + E_hard * (strain - eps_y)
        sigma_hard_neg = -self.fy + E_hard * (strain + eps_y)

        sigma = torch.where(
            strain.abs() <= eps_y,
            sigma_elastic,
            torch.where(strain > 0, sigma_hard_pos, sigma_hard_neg),
        )

        # Clamp beyond ultimate strain
        rupt = strain.abs() > self.eps_u
        sigma = torch.where(rupt, torch.zeros_like(sigma), sigma)

        return sigma

    def tangent(self, strain: torch.Tensor) -> torch.Tensor:
        """Tangent modulus."""
        eps_y = self.eps_y
        E_hard = self.b * self.Es

        Et = torch.where(
            strain.abs() <= eps_y,
            torch.full_like(strain, self.Es),
            torch.full_like(strain, E_hard),
        )
        rupt = strain.abs() > self.eps_u
        Et = torch.where(rupt, torch.zeros_like(Et), Et)
        return Et

    def get_parameters(self) -> Dict[str, float]:
        return {
            "fy": self.fy,
            "Es": self.Es,
            "b": self.b,
            "eps_u": self.eps_u,
        }

    def set_parameters(self, **kwargs: float) -> None:
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, val)
