"""Forward analysis solver — builds and trains a PINN for known parameters."""

from typing import Dict

import torch

from configs.base_config import BeamConfig
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from utils import get_device, set_seed, uniform_collocation, boundary_points
from .trainer import Trainer


def _extract_fiber_geometry(rc: RCRectSection):
    fibers = rc.section.fibers
    y = fibers.positions()
    A = fibers.areas()
    is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]
    return y, A, is_steel


def _compute_EI(rc: RCRectSection) -> float:
    """EI = sum(E_i * A_i * y_i^2)."""
    EI = 0.0
    for f in rc.section.fibers.fibers:
        E = f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec
        EI += E * f.area * f.y ** 2
    return EI


class ForwardSolver:
    """Set up and run a forward PINN beam analysis from a BeamConfig."""

    def __init__(self, config: BeamConfig, log_dir: str | None = None):
        self.config = config
        self.device = get_device()
        self.log_dir = log_dir

    def solve(self) -> Dict:
        cfg = self.config
        set_seed(cfg.seed)

        # --- materials ---
        if cfg.elastic:
            concrete_eff = ManderConcrete(
                fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2,
                Gf=cfg.Gf, h=cfg.concrete_h,
            )
            steel_eff = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
        else:
            concrete_eff = ManderConcrete(
                fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co, eps_cu=cfg.eps_cu,
                Gf=cfg.Gf, h=cfg.concrete_h,
            )
            steel_eff = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)

        # --- section ---
        rc = RCRectSection(
            width=cfg.section_width, height=cfg.section_height,
            concrete=concrete_eff, steel=steel_eff,
            n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
        )

        # --- non-dim scales ---
        scales = NonDimScales(
            L=cfg.beam_length, E_ref=cfg.Ec,
            A_ref=rc.gross_area, I_ref=rc.gross_inertia,
        )

        # --- normalization coefficients ---
        EI = _compute_EI(rc)
        nc = scales.norm_coeffs(cfg.q, EI)
        print(f"  Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, "
              f"eps0={nc['eps0']:.3e}, N={nc['N']:.3e}")

        # --- fiber geometry ---
        fibers_y, fibers_A, fibers_is_steel = _extract_fiber_geometry(rc)

        # --- PINN ---
        field_nets = FieldNetworks(
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            norm_coeffs=nc,
        ).to(self.device)

        pinn = PINNBeamModel(
            field_nets=field_nets, section=rc.section, scales=scales,
            loss_weights=cfg.loss_weights, elastic=cfg.elastic,
            fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
        )

        # --- collocation ---
        xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=self.device)
        xi_bc = boundary_points(0.0, 1.0, device=self.device)
        q_bar = scales.to_nondim_q(cfg.q)

        # --- optimizer ---
        optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
        trainer = Trainer(pinn, optimizer, scheduler, log_dir=self.log_dir)

        logger = trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs)

        return {
            "model": pinn, "field_nets": field_nets,
            "logger": logger, "scales": scales, "config": cfg,
        }
