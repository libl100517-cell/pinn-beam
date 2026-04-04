"""Loss assembly for PINN training.

Combines physics residual losses, boundary condition losses, and data losses
into a single weighted total loss.
"""

from typing import Dict, Tuple

import torch


class PINNLoss:
    """Weighted loss assembler.

    Parameters
    ----------
    weights : dict
        Mapping from loss-component name to weight.
        Example: {"physics": 1.0, "bc": 10.0, "data_disp": 1.0}
    """

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {
            "equil_M": 1.0,
            "equil_N": 1e5,
            "const_M": 1.0,
            "bc": 10.0,
            "M_net_bc": 10.0,
            "M_sec_bc": 10.0,
            "N_sec_bc": 1e5,
            "data_disp": 1.0,
        }

    def physics_loss(self, residuals: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MSE over physics residuals.

        Parameters
        ----------
        residuals : dict
            Keys like "compat", "equil_M", etc.  Values are residual tensors.

        Returns
        -------
        total : scalar tensor
        components : dict of scalar tensors per residual type
        """
        components = {}
        total = torch.tensor(0.0, device=next(iter(residuals.values())).device)
        for name, res in residuals.items():
            mse = (res ** 2).mean()
            w = self.weights.get(name, 1.0)
            components[name] = mse
            total = total + w * mse
        return total, components

    def bc_loss(self, bc_residuals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE over boundary condition residuals."""
        total = torch.tensor(0.0, device=next(iter(bc_residuals.values())).device)
        for res in bc_residuals.values():
            total = total + (res ** 2).mean()
        return self.weights.get("bc", 10.0) * total

    def data_displacement_loss(
        self,
        w_pred: torch.Tensor,
        w_obs: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and observed displacements."""
        return self.weights.get("data_disp", 1.0) * ((w_pred - w_obs) ** 2).mean()

    def data_crack_width_loss(
        self,
        crack_pred: torch.Tensor,
        crack_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Placeholder for crack width data loss.

        This method provides the extension point for future crack width
        observation models.  Currently raises NotImplementedError so that
        it is explicitly clear this needs a crack-response model.
        """
        return self.weights.get("data_crack", 1.0) * ((crack_pred - crack_obs) ** 2).mean()
