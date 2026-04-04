"""Inverse analysis solver — identifies material parameters from observations."""

from typing import Dict

import torch

from configs.base_config import BeamConfig
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel, InverseParameterRegistry
from utils import get_device, set_seed, uniform_collocation, boundary_points
from .trainer import Trainer


def _extract_fiber_geometry(rc: RCRectSection):
    """Extract fiber positions, areas, and steel flags for differentiable path."""
    fibers = rc.section.fibers
    y = fibers.positions()
    A = fibers.areas()
    is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]
    return y, A, is_steel


class InverseSolver:
    """Set up and run an inverse PINN analysis from a BeamConfig.

    Parameters
    ----------
    config : BeamConfig
        Problem configuration.
    trainable_params : dict
        Parameters to identify.  Keys are parameter names ("Ec", "Es", "fc", "fy").
        Values are dicts with "init", "bounds" (optional).
        Example: {"Ec": {"init": 20000.0, "bounds": (10000, 50000)}}
    observation_xi : torch.Tensor
        Non-dimensional observation locations, shape (n_obs, 1).
    observation_w : torch.Tensor
        Non-dimensional observed displacements, shape (n_obs, 1).
    """

    def __init__(
        self,
        config: BeamConfig,
        trainable_params: Dict[str, Dict],
        observation_xi: torch.Tensor,
        observation_w: torch.Tensor,
    ):
        self.config = config
        self.trainable_params = trainable_params
        self.observation_xi = observation_xi
        self.observation_w = observation_w
        self.device = get_device()

    def solve(self) -> Dict:
        """Run inverse analysis."""
        cfg = self.config
        set_seed(cfg.seed)

        # --- materials (initial guesses) ---
        if cfg.elastic:
            concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2)
            steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
        else:
            concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co, eps_cu=cfg.eps_cu)
            steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)

        # --- section ---
        rc = RCRectSection(
            width=cfg.section_width,
            height=cfg.section_height,
            concrete=concrete,
            steel=steel,
            n_concrete_fibers=cfg.n_concrete_fibers,
            rebar_layout=cfg.rebar_layout,
        )

        # --- non-dim scales (use config values as reference) ---
        scales = NonDimScales(
            L=cfg.beam_length,
            E_ref=cfg.Ec,
            A_ref=rc.gross_area,
            I_ref=rc.gross_inertia,
        )

        # --- inverse parameter registry ---
        registry = InverseParameterRegistry()
        for name, spec in self.trainable_params.items():
            registry.register(
                name,
                init_value=spec["init"],
                bounds=spec.get("bounds"),
                trainable=True,
            )
        registry = registry.to(self.device)

        # --- fiber geometry for differentiable path ---
        fibers_y, fibers_A, fibers_is_steel = _extract_fiber_geometry(rc)

        # --- PINN ---
        field_nets = FieldNetworks(
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
        ).to(self.device)

        pinn = PINNBeamModel(
            field_nets=field_nets,
            section=rc.section,
            scales=scales,
            loss_weights=cfg.loss_weights,
            inverse_registry=registry,
            elastic=cfg.elastic,
            fibers_y=fibers_y,
            fibers_A=fibers_A,
            fibers_is_steel=fibers_is_steel,
        )

        # --- collocation ---
        xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=self.device)
        xi_bc = boundary_points(0.0, 1.0, device=self.device)
        q_bar = scales.to_nondim_q(cfg.q)

        xi_data = self.observation_xi.to(self.device)
        w_data = self.observation_w.to(self.device)

        # --- optimizer (separate param groups: higher lr for inverse params) ---
        optimizer = torch.optim.Adam([
            {"params": field_nets.parameters(), "lr": cfg.learning_rate},
            {"params": registry.parameters(), "lr": cfg.learning_rate * 10},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
        trainer = Trainer(pinn, optimizer, scheduler)

        logger = trainer.train(
            xi_col, xi_bc, q_bar, cfg.n_epochs,
            xi_data=xi_data, w_data=w_data,
        )

        return {
            "model": pinn,
            "field_nets": field_nets,
            "registry": registry,
            "logger": logger,
            "scales": scales,
            "config": cfg,
            "identified_params": registry.get_values(),
        }
