"""PINN beam model: coordinates field networks, section, and physics.

Loss structure (no N network):
    1. w̄ 求二阶导 → κ̄ = -d²w̄/dξ²
    2. (ε̄₀, κ̄) → 纤维截面积分 → N_sec, M_sec
    3. 本构 loss: M̄_net vs M̄_sec
    4. 平衡 loss: d²M̄_net/dξ² / (q̄·C_eq) + 1 = 0
    5. 轴力平衡: dN̄_sec/dξ = 0 (梯度穿过截面积分回传 eps0 和 w)
    6. BC loss: w̄(0)=w̄(1)=0, M̄(0)=M̄(1)=0, N_sec(0)=N_sec(1)=N_applied
    7. 数据 loss: w̄_pred vs w̄_obs (inverse only)
"""

from typing import Dict, Tuple

import torch

from physics.nondimensional import NonDimScales
from physics.beam_equations import BeamEquations
from physics.boundary_conditions import SSBeamBC
from physics.losses import PINNLoss
from sections.fiber_section import FiberSection
from .field_nets import FieldNetworks
from .inverse_parameters import InverseParameterRegistry


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


class PINNBeamModel:
    """High-level PINN beam coordinator."""

    def __init__(
        self,
        field_nets: FieldNetworks,
        section: FiberSection,
        scales: NonDimScales,
        loss_weights: Dict[str, float] | None = None,
        inverse_registry: InverseParameterRegistry | None = None,
        elastic: bool = True,
        fibers_y: torch.Tensor | None = None,
        fibers_A: torch.Tensor | None = None,
        fibers_is_steel: list | None = None,
        N_applied: float = 0.0,
        norm_coeffs: Dict[str, float] | None = None,
    ):
        self.field_nets = field_nets
        self.section = section
        self.scales = scales
        self.beam_eq = BeamEquations(scales)
        self.bc = SSBeamBC()
        self.loss_fn = PINNLoss(loss_weights)
        self.inverse_registry = inverse_registry
        self.elastic = elastic
        self.fibers_y = fibers_y
        self.fibers_A = fibers_A
        self.fibers_is_steel = fibers_is_steel
        self.N_applied_bar = N_applied / scales.F_ref

    # ------------------------------------------------------------------
    # Section response — all on GPU
    # ------------------------------------------------------------------

    def _elastic_section_response(
        self,
        eps0: torch.Tensor,
        kappa: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable elastic fiber section: σ = E·ε, then integrate."""
        inv = self.inverse_registry.get_dict() if self.inverse_registry else {}

        Ec_val = inv.get("Ec", torch.tensor(self.scales.E_ref, device=device))
        Es_val = inv.get("Es", torch.tensor(200000.0, device=device))
        if not isinstance(Ec_val, torch.Tensor):
            Ec_val = torch.tensor(float(Ec_val), device=device)
        if not isinstance(Es_val, torch.Tensor):
            Es_val = torch.tensor(float(Es_val), device=device)
        Ec_val, Es_val = Ec_val.to(device), Es_val.to(device)

        y = self.fibers_y.to(device)
        A = self.fibers_A.to(device)
        fibers_E = torch.stack([Es_val if s else Ec_val for s in self.fibers_is_steel])

        strain = eps0 - kappa * y.unsqueeze(0)
        stress = strain * fibers_E.unsqueeze(0)
        N = (stress * A.unsqueeze(0)).sum(dim=1, keepdim=True)
        M = (stress * A.unsqueeze(0) * (-y.unsqueeze(0))).sum(dim=1, keepdim=True)
        return N, M

    def _nonlinear_section_response(
        self,
        eps0: torch.Tensor,
        kappa: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inverse_registry is not None:
            vals = self.inverse_registry.get_dict()
            for fiber in self.section.fibers.fibers:
                mat = fiber.material
                params = mat.get_parameters()
                update = {k: vals[k].item() for k in params if k in vals}
                if update:
                    mat.set_parameters(**update)
        sec_resp = self.section.response(eps0, kappa, device=device)
        return sec_resp["N"], sec_resp["M"]

    def _section_response(self, eps0_dim, kappa_dim, device):
        """Dispatch to elastic or nonlinear section response."""
        if self.elastic and self.fibers_y is not None:
            return self._elastic_section_response(eps0_dim, kappa_dim, device)
        else:
            return self._nonlinear_section_response(eps0_dim, kappa_dim, device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        xi_col: torch.Tensor,
        xi_bc: torch.Tensor,
        q_bar: float,
        xi_data: torch.Tensor | None = None,
        w_data: torch.Tensor | None = None,
        adaptive_weights: Dict[str, float] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        device = xi_col.device
        aw = adaptive_weights or {}

        # === 三个网络前向 (w, eps0, M) ===
        fields = self.field_nets(xi_col)
        w_bar = fields["w_bar"]
        eps0_bar = fields["eps0_bar"]
        M_bar_net = fields["M_bar"]

        # === w̄ 求二阶导 → 曲率 ===
        dw = _grad(w_bar, xi_col)
        d2w = _grad(dw, xi_col)
        kappa_bar = -d2w

        # === 纤维截面积分 → N_sec, M_sec ===
        eps0_dim = eps0_bar * self.scales.eps_ref
        kappa_dim = kappa_bar * self.scales.kap_ref
        N_sec_dim, M_sec_dim = self._section_response(eps0_dim, kappa_dim, device)

        M_bar_sec = self.scales.to_nondim_M(M_sec_dim)
        N_bar_sec = self.scales.to_nondim_N(N_sec_dim)

        # === 本构 loss: M_net vs M_sec ===
        const_M_ptw = (M_bar_net - M_bar_sec) ** 2

        raw_losses = {
            "const_M": const_M_ptw.mean(),
        }

        # === 平衡 loss: M ===
        dM = _grad(M_bar_net, xi_col)
        d2M = _grad(dM, xi_col)
        C_eq = self.scales.A_ref * self.scales.L ** 2 / self.scales.I_ref
        equil_M_res = d2M / (q_bar * C_eq) + 1.0
        equil_M_ptw = equil_M_res ** 2
        raw_losses["equil_M"] = equil_M_ptw.mean()

        # === 轴力平衡: dN_sec/dξ = 0 ===
        # N_sec 来自截面积分，梯度穿过截面回传到 eps0 和 w 网络
        dN_sec = _grad(N_bar_sec, xi_col)
        equil_N_ptw = dN_sec ** 2
        raw_losses["equil_N"] = equil_N_ptw.mean()

        # 逐点残差 (detach，仅用于采样)
        pointwise_residual = (const_M_ptw + equil_M_ptw + equil_N_ptw).detach()

        # === BC loss ===
        bc_xi = xi_bc.detach().requires_grad_(True)
        bc_fields = self.field_nets(bc_xi)

        # w 边界: w(0)=w(1)=0
        raw_losses["bc"] = (bc_fields["w_bar"][0:1] ** 2).mean() + \
                           (bc_fields["w_bar"][1:2] ** 2).mean()

        # M 网络边界: M_net(0)=M_net(1)=0
        raw_losses["M_net_bc"] = (bc_fields["M_bar"][0:1] ** 2).mean() + \
                                  (bc_fields["M_bar"][1:2] ** 2).mean()

        # 纤维截面边界 (κ=0 at supports)
        N_sec_bc_0, M_sec_bc_0 = self._section_response(
            bc_fields["eps0_bar"][0:1] * self.scales.eps_ref,
            torch.zeros(1, 1, device=device), device,
        )
        N_sec_bc_1, M_sec_bc_1 = self._section_response(
            bc_fields["eps0_bar"][1:2] * self.scales.eps_ref,
            torch.zeros(1, 1, device=device), device,
        )

        # M_sec 边界: M_sec(0)=M_sec(1)=0
        M_bc_bar_0 = self.scales.to_nondim_M(M_sec_bc_0)
        M_bc_bar_1 = self.scales.to_nondim_M(M_sec_bc_1)
        raw_losses["M_sec_bc"] = (M_bc_bar_0 ** 2).mean() + (M_bc_bar_1 ** 2).mean()

        # N_sec 边界: N_sec = N_applied
        N_bc_bar_0 = self.scales.to_nondim_N(N_sec_bc_0)
        N_bc_bar_1 = self.scales.to_nondim_N(N_sec_bc_1)
        raw_losses["N_sec_bc"] = ((N_bc_bar_0 - self.N_applied_bar) ** 2).mean() + \
                                  ((N_bc_bar_1 - self.N_applied_bar) ** 2).mean()

        # === 数据 loss (inverse only) ===
        if xi_data is not None and w_data is not None:
            w_pred = self.field_nets.net_w(xi_data)
            raw_losses["data_disp"] = ((w_pred - w_data) ** 2).mean()

        # === 加权求总 loss ===
        total = torch.tensor(0.0, device=device)
        for name, raw in raw_losses.items():
            w_manual = self.loss_fn.weights.get(name, 1.0)
            w_ntk = aw.get(name, 1.0)
            total = total + w_manual * w_ntk * raw

        components = {name: raw.item() for name, raw in raw_losses.items()}
        components["total"] = total.item()
        return total, components, raw_losses, pointwise_residual

    def predict(self, xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict all fields. Returns CPU tensors."""
        device = next(self.field_nets.parameters()).device
        xi_dev = xi.to(device)
        with torch.no_grad():
            fields = self.field_nets(xi_dev)
        return {k: v.cpu() for k, v in fields.items()}
