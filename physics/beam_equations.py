"""Beam governing equations in non-dimensional form.

Euler-Bernoulli beam relations (dimensional):
    κ  = -w''                       (compatibility)
    M  = ∫ σ(ε) y dA               (constitutive / fiber resultant)
    M'' = -q(x)                     (equilibrium, transverse)
    N'  = 0                         (equilibrium, axial — no distributed axial load)

Non-dimensional forms (barred quantities):
    κ̄  = -d²w̄/dξ²  ·  (w_ref / (L² · κ_ref))
    equilibrium:  d²M̄/dξ² = -q̄  ·  (q_ref · L² / M_ref)
    axial equil:  dN̄/dξ   = 0

The coefficient (q_ref · L² / M_ref) equals (E_ref·A_ref·L / (E_ref·I_ref/L)) · (L²/L)
= A_ref·L² / I_ref   — a pure geometric ratio.

This module provides residual functions that return the PDE residuals at
collocation points.  All inputs and outputs are non-dimensional.
"""

from typing import Dict

import torch

from .nondimensional import NonDimScales


class BeamEquations:
    """Compute PDE residuals for Euler-Bernoulli beam (non-dimensional).

    Parameters
    ----------
    scales : NonDimScales
        Reference scales for non-dimensionalization.
    """

    def __init__(self, scales: NonDimScales):
        self.scales = scales

    # ------------------------------------------------------------------
    # Automatic-differentiation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """dy/dx via autograd (x must have requires_grad)."""
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True,
        )[0]

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def compatibility_residual(
        self,
        xi: torch.Tensor,
        w_bar: torch.Tensor,
        kappa_bar_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility: κ̄ + d²w̄/dξ² · C_compat = 0.

        Where C_compat = w_ref / (L² · kap_ref) = L / (L² · (1/L)) = 1.
        So the relation simplifies to:  κ̄ + d²w̄/dξ² = 0.
        """
        dw = self._grad(w_bar, xi)
        d2w = self._grad(dw, xi)
        return kappa_bar_pred + d2w

    def equilibrium_residual(
        self,
        xi: torch.Tensor,
        M_bar: torch.Tensor,
        q_bar: float,
    ) -> torch.Tensor:
        """Transverse equilibrium: d²M̄/dξ² + q̄ · C_eq = 0.

        C_eq = q_ref · L² / M_ref = A_ref · L² / I_ref.
        """
        dM = self._grad(M_bar, xi)
        d2M = self._grad(dM, xi)
        C_eq = self.scales.A_ref * self.scales.L ** 2 / self.scales.I_ref
        return d2M + q_bar * C_eq

    def axial_equilibrium_residual(
        self,
        xi: torch.Tensor,
        N_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Axial equilibrium: dN̄/dξ = 0."""
        dN = self._grad(N_bar, xi)
        return dN

    def constitutive_residual(
        self,
        M_bar_net: torch.Tensor,
        M_bar_section: torch.Tensor,
    ) -> torch.Tensor:
        """Section consistency: M̄_net − M̄_section = 0."""
        return M_bar_net - M_bar_section

    def axial_constitutive_residual(
        self,
        N_bar_net: torch.Tensor,
        N_bar_section: torch.Tensor,
    ) -> torch.Tensor:
        """Axial section consistency: N̄_net − N̄_section = 0."""
        return N_bar_net - N_bar_section

    def all_residuals(
        self,
        xi: torch.Tensor,
        w_bar: torch.Tensor,
        eps0_bar: torch.Tensor,
        M_bar_net: torch.Tensor,
        N_bar_net: torch.Tensor,
        M_bar_section: torch.Tensor,
        N_bar_section: torch.Tensor,
        kappa_bar: torch.Tensor,
        q_bar: float,
    ) -> Dict[str, torch.Tensor]:
        """Return all PDE residuals as a dict."""
        return {
            "compat": self.compatibility_residual(xi, w_bar, kappa_bar),
            "equil_M": self.equilibrium_residual(xi, M_bar_net, q_bar),
            "equil_N": self.axial_equilibrium_residual(xi, N_bar_net),
            "const_M": self.constitutive_residual(M_bar_net, M_bar_section),
            "const_N": self.axial_constitutive_residual(N_bar_net, N_bar_section),
        }
