"""Base configuration dataclass for PINN beam analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BeamConfig:
    """Full configuration for a PINN beam problem."""

    # --- geometry ---
    beam_length: float = 3000.0        # mm
    section_width: float = 1000.0       # mm
    section_height: float = 120.0      # mm

    # --- concrete (Mander) ---
    fc: float = 30.0                   # MPa
    Ec: float = 30000.0                # MPa
    eps_co: float = -0.002
    eps_cu: float = -0.0033
    Gf: float = 0.1                    # N/mm  fracture energy
    concrete_h: float = 150.0          # mm    crack band width

    # --- steel (bilinear) ---
    fy: float = 335.0                  # MPa
    Es: float = 200000.0              # MPa
    steel_b: float = 0.01             # hardening ratio

    # --- reinforcement layout: list of (y_from_centroid_mm, area_mm2) ---
    rebar_layout: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-35.0, 2011.0),   # bottom bars: 10ϕ16 ≈ 10×201
    ])
    n_concrete_fibers: int = 20

    # --- loading ---
    q: float = 20.0                    # N/mm  uniform distributed load
    N_applied: float = 0.0             # N     applied axial force (+ tension)

    # --- PINN architecture ---
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "tanh"
    use_fourier: bool = False
    n_frequencies: int = 16
    fourier_sigma: float = 1.0

    # --- training ---
    n_collocation: int = 50
    n_epochs: int = 5000
    learning_rate: float = 1e-3
    seed: int = 42

    # --- loss weights ---
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "equil_M": 100.0,
        "equil_N": 1e3,
        "const_M": 10.0,
        "bc": 100.0,
        "M_net_bc": 100.0,
        "M_sec_bc": 100.0,
        "N_sec_bc": 1e4,
        "data_disp": 1.0,
    })

    # --- analysis mode ---
    mode: str = "forward"              # "forward" or "inverse"
    elastic: bool = True               # True = elastic, False = elasto-plastic
