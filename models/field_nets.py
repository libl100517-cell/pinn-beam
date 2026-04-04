"""Three independent MLPs for the structural state fields.

Each network outputs O(1) values. The final non-dimensional field is:

    field_bar = net(xi) * norm_coeff

norm_coeff controls the output range so the network doesn't need to
learn very small or very large values.

N is not a network output — it is computed from the fiber section
response using eps0 and kappa (from w'').
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .mlp import MLP


class FieldNetworks(nn.Module):
    """Wrapper around three independent field MLPs with normalization coefficients.

    Parameters
    ----------
    hidden_dims : list of int
    activation : str
    norm_coeffs : dict, optional
        {"w": float, "M": float, "eps0": float}.
        Each network's raw output is multiplied by its coefficient.
        Default all 1.0 (no normalization).
    """

    def __init__(
        self,
        hidden_dims: List[int] | None = None,
        activation: str = "tanh",
        norm_coeffs: Dict[str, float] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 32, 32]

        self.net_w = MLP(1, 1, hidden_dims, activation)
        self.net_eps0 = MLP(1, 1, hidden_dims, activation)
        self.net_M = MLP(1, 1, hidden_dims, activation)

        c = norm_coeffs or {}
        self.register_buffer("c_w",    torch.tensor(c.get("w", 1.0)))
        self.register_buffer("c_eps0", torch.tensor(c.get("eps0", 1.0)))
        self.register_buffer("c_M",    torch.tensor(c.get("M", 1.0)))

    def forward(self, xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "w_bar":    self.net_w(xi)    * self.c_w,
            "eps0_bar": self.net_eps0(xi) * self.c_eps0,
            "M_bar":    self.net_M(xi)    * self.c_M,
        }
