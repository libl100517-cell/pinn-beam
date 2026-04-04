"""Fiber section integration: compute resultant N and M from fiber stresses.

Given a centroidal strain eps0 and curvature kappa, each fiber's strain is:
    eps_i = eps0 - y_i * kappa

The section resultants are:
    N = sum( sigma_i * A_i )
    M = sum( sigma_i * A_i * (-y_i) )

(positive M produces tension on the bottom fiber, consistent with standard
beam convention where y is measured upward from the centroid).
"""

from typing import Dict, Tuple

import torch

from .fibers import FiberCollection


class FiberSection:
    """Compute section force resultants via fiber integration.

    Parameters
    ----------
    fibers : FiberCollection
        The fibers that discretise the section.
    """

    def __init__(self, fibers: FiberCollection):
        self.fibers = fibers

    def response(
        self,
        eps0: torch.Tensor,
        kappa: torch.Tensor,
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute N, M, and per-fiber stresses.

        Parameters
        ----------
        eps0 : (n_pts,) or (n_pts, 1)
            Centroidal axial strain at each evaluation point.
        kappa : (n_pts,) or (n_pts, 1)
            Curvature at each evaluation point.

        Returns
        -------
        dict with keys:
            N      : (n_pts, 1) axial force resultant
            M      : (n_pts, 1) bending moment resultant
            strains: (n_pts, n_fibers) fiber strains
            stresses: (n_pts, n_fibers) fiber stresses
        """
        eps0 = eps0.reshape(-1, 1)
        kappa = kappa.reshape(-1, 1)

        y = self.fibers.positions(device=device)   # (n_fibers,)
        A = self.fibers.areas(device=device)        # (n_fibers,)

        # Fiber strains: eps_i = eps0 - y_i * kappa
        fiber_strains = eps0 - kappa * y.unsqueeze(0)  # (n_pts, n_fibers)

        # Evaluate stress per fiber using its own material
        fiber_stresses = torch.zeros_like(fiber_strains)
        for i, fiber in enumerate(self.fibers.fibers):
            fiber_stresses[:, i] = fiber.material.stress(fiber_strains[:, i])

        # Resultants
        N = (fiber_stresses * A.unsqueeze(0)).sum(dim=1, keepdim=True)
        M = (fiber_stresses * A.unsqueeze(0) * (-y.unsqueeze(0))).sum(dim=1, keepdim=True)

        return {
            "N": N,
            "M": M,
            "strains": fiber_strains,
            "stresses": fiber_stresses,
        }

    def tangent(
        self,
        eps0: torch.Tensor,
        kappa: torch.Tensor,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Section tangent stiffness [EA, 0; 0, EI] (uncoupled for symmetric section).

        Returns EA, ES, EI tensors of shape (n_pts, 1).
        """
        eps0 = eps0.reshape(-1, 1)
        kappa = kappa.reshape(-1, 1)

        y = self.fibers.positions(device=device)
        A = self.fibers.areas(device=device)

        fiber_strains = eps0 - kappa * y.unsqueeze(0)

        Et = torch.zeros_like(fiber_strains)
        for i, fiber in enumerate(self.fibers.fibers):
            Et[:, i] = fiber.material.tangent(fiber_strains[:, i])

        EA = (Et * A.unsqueeze(0)).sum(dim=1, keepdim=True)
        ES = (Et * A.unsqueeze(0) * y.unsqueeze(0)).sum(dim=1, keepdim=True)
        EI = (Et * A.unsqueeze(0) * (y.unsqueeze(0) ** 2)).sum(dim=1, keepdim=True)

        return EA, ES, EI
