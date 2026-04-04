"""Boundary condition modules.

Simply supported beam (Euler-Bernoulli):
    w(0) = 0,  w(L) = 0     — zero displacement at supports
    M(0) = 0,  M(L) = 0     — zero moment at supports

In non-dimensional form:
    w̄(0) = 0,  w̄(1) = 0
    M̄(0) = 0,  M̄(1) = 0
"""

from typing import Callable, Dict, List

import torch


class SSBeamBC:
    """Simply supported beam boundary conditions (non-dimensional).

    Evaluates BC residuals given the PINN output at ξ = 0 and ξ = 1.
    """

    def residuals(
        self,
        w_bar_0: torch.Tensor,
        w_bar_1: torch.Tensor,
        M_bar_0: torch.Tensor,
        M_bar_1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "w_0": w_bar_0,
            "w_1": w_bar_1,
            "M_0": M_bar_0,
            "M_1": M_bar_1,
        }
