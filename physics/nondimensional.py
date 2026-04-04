"""Non-dimensionalization module.

All governing equations are solved in dimensionless form.  This module defines
the reference scales and provides helpers to convert between dimensional and
non-dimensional quantities.

Reference scales
----------------
L_ref   : beam length  (length)
F_ref   : reference force, e.g. E_c * A  (force)
M_ref   : reference moment, e.g. E_c * I  / L_ref  or F_ref * L_ref  (force·length)
w_ref   : reference displacement = L_ref  (length)
eps_ref : reference strain = 1  (dimensionless already, but kept for clarity)
kap_ref : reference curvature = 1 / L_ref  (1/length)
q_ref   : reference distributed load = F_ref / L_ref  (force/length)

Dimensionless variables
-----------------------
ξ  = x / L_ref          ∈ [0, 1]
w̄  = w / w_ref
ε̄₀ = eps0 / eps_ref
M̄  = M / M_ref
N̄  = N / F_ref
q̄  = q / q_ref
κ̄  = κ / kap_ref
"""

from dataclasses import dataclass

import torch


@dataclass
class NonDimScales:
    """Container for non-dimensionalization reference scales.

    Parameters
    ----------
    L : float
        Beam length (mm).
    E_ref : float
        Reference elastic modulus (MPa).
    A_ref : float
        Reference cross-section area (mm^2).
    I_ref : float
        Reference second moment of area (mm^4).
    """
    L: float
    E_ref: float
    A_ref: float
    I_ref: float

    # --- derived scales ---

    @property
    def L_ref(self) -> float:
        return self.L

    @property
    def F_ref(self) -> float:
        """Reference force = E_ref * A_ref."""
        return self.E_ref * self.A_ref

    @property
    def M_ref(self) -> float:
        """Reference moment = E_ref * I_ref / L_ref."""
        return self.E_ref * self.I_ref / self.L

    @property
    def w_ref(self) -> float:
        """Reference displacement = L."""
        return self.L

    @property
    def eps_ref(self) -> float:
        return 1.0

    @property
    def kap_ref(self) -> float:
        """Reference curvature = 1 / L."""
        return 1.0 / self.L

    @property
    def q_ref(self) -> float:
        """Reference distributed load = F_ref / L_ref."""
        return self.F_ref / self.L

    # --- conversion helpers ---

    def to_nondim_x(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.L_ref

    def to_dim_x(self, xi: torch.Tensor) -> torch.Tensor:
        return xi * self.L_ref

    def to_nondim_w(self, w: torch.Tensor) -> torch.Tensor:
        return w / self.w_ref

    def to_dim_w(self, w_bar: torch.Tensor) -> torch.Tensor:
        return w_bar * self.w_ref

    def to_nondim_M(self, M: torch.Tensor) -> torch.Tensor:
        return M / self.M_ref

    def to_dim_M(self, M_bar: torch.Tensor) -> torch.Tensor:
        return M_bar * self.M_ref

    def to_nondim_N(self, N: torch.Tensor) -> torch.Tensor:
        return N / self.F_ref

    def to_dim_N(self, N_bar: torch.Tensor) -> torch.Tensor:
        return N_bar * self.F_ref

    def to_nondim_q(self, q: float) -> float:
        return q / self.q_ref

    def to_dim_q(self, q_bar: float) -> float:
        return q_bar * self.q_ref

    def to_nondim_kappa(self, kappa: torch.Tensor) -> torch.Tensor:
        return kappa / self.kap_ref

    def to_dim_kappa(self, kappa_bar: torch.Tensor) -> torch.Tensor:
        return kappa_bar * self.kap_ref

    # --- output normalization scales ---

    def norm_coeffs(self, q: float, EI_eff: float) -> dict:
        """Compute normalization coefficients so each network outputs O(1).

        final_nondim_value = net_output_O1 * norm_coeff

        Parameters
        ----------
        q : float
            Distributed load (N/mm).
        EI_eff : float
            Effective bending stiffness at N=0 (N·mm²).
        """
        # Expected order of magnitude for each non-dim field
        w_mid = 5 * q * self.L ** 4 / (384 * EI_eff)   # mm, expected midspan w
        M_mid = q * self.L ** 2 / 8                      # N·mm, expected midspan M

        return {
            "w":    w_mid / self.w_ref,                  # ~7e-5
            "M":    M_mid / self.M_ref,                  # ~9e-4
            "eps0": 3300e-6 / self.eps_ref,              # 3300με reference
            "N":    3300e-6 * self.E_ref * self.A_ref / self.F_ref,  # EA·ε_typical / F_ref ≈ 3.3e-3
        }
