#!/usr/bin/env python3
"""Quick sweep for nonlinear PINN tuning (5000 epochs each).

Tests: const_M weight, learning rate, alpha_ts.

Usage:
    cd pinn
    python -m examples.sweep_nonlinear
"""

import os, sys, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.base_config import BeamConfig
from materials import SmoothConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer
from solvers.section_analysis import solve_beam_nonlinear


def compute_section_stiffness(rc):
    EA, ES, EI = 0.0, 0.0, 0.0
    for f in rc.section.fibers.fibers:
        E = f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec
        EA += E * f.area; ES += E * f.area * f.y; EI += E * f.area * f.y ** 2
    return EA, ES, EI


def run_one(cfg, concrete, steel, device, label=""):
    set_seed(cfg.seed)
    L, q = cfg.beam_length, cfg.q

    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    scales = NonDimScales(L=L, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)
    nc = scales.norm_coeffs(q, EI)

    fibers = rc.section.fibers
    field_nets = FieldNetworks(hidden_dims=cfg.hidden_dims, activation=cfg.activation, norm_coeffs=nc).to(device)
    pinn = PINNBeamModel(
        field_nets=field_nets, section=rc.section, scales=scales,
        loss_weights=cfg.loss_weights, elastic=False,
        fibers_y=fibers.positions(), fibers_A=fibers.areas(),
        fibers_is_steel=[isinstance(f.material, BilinearSteel) for f in fibers.fibers],
        N_applied=cfg.N_applied, norm_coeffs=nc,
    )

    xi_col = uniform_collocation(cfg.n_collocation, device=device)
    xi_bc = boundary_points(device=device)
    q_bar = scales.to_nondim_q(q)
    optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    trainer = Trainer(pinn, optimizer, scheduler, use_ntk=False, resample_every=500)
    trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs, print_every=cfg.n_epochs + 1)

    # Reference
    M_mid = q * L**2 / 8
    ref = solve_beam_nonlinear(rc.section, L, q, n_pts=200, kappa_max=M_mid/EI*5)

    # Predict
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    xi_dev = xi_plot.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)
    dw = _grad(fields["w_bar"], xi_dev)
    d2w = _grad(dw, xi_dev)
    kappa_bar = -d2w
    N_sec, M_sec = pinn._section_response(
        fields["eps0_bar"] * scales.eps_ref, kappa_bar * scales.kap_ref, device)
    x = xi_plot.numpy().flatten() * L
    with torch.no_grad():
        w = fields["w_bar"].cpu().numpy().flatten() * scales.w_ref
        M_net = fields["M_bar"].cpu().numpy().flatten() * scales.M_ref
        N_sec_np = N_sec.cpu().numpy().flatten()

    from scipy.interpolate import interp1d
    ref_w = interp1d(ref["x"], ref["w"], kind="cubic")(x)
    ref_N = interp1d(ref["x"], ref["N"], kind="cubic")(x)

    nrmse_w = np.sqrt(np.mean((w - ref_w)**2)) / np.ptp(ref_w) * 100
    rmse_N = np.sqrt(np.mean((N_sec_np - ref_N)**2))

    return {"label": label, "nrmse_w": nrmse_w, "rmse_N": rmse_N,
            "w_mid": w[100], "w_ref": ref["w"][100]}


def main():
    device = get_device()
    base_cfg = BeamConfig(
        mode="forward", elastic=False, n_epochs=5000, learning_rate=1e-4,
        n_collocation=200, activation="tanh", q=40.0, N_applied=0.0,
    )
    fc, Ec, eps_co = base_cfg.fc, base_cfg.Ec, base_cfg.eps_co
    ft_default = 0.62 * fc ** 0.5

    # Steel with tension stiffening reduction
    h = base_cfg.section_height
    d_rebar = abs(base_cfg.rebar_layout[0][0])
    c_bottom = h / 2 - d_rebar
    h_eff = min(2.5 * c_bottom, h / 2)
    Ac_eff = base_cfg.section_width * h_eff
    As = sum(a for _, a in base_cfg.rebar_layout)
    delta_fy = ft_default * Ac_eff / As
    fy_embedded = base_cfg.fy - delta_fy

    # Sweep parameters
    tests = []

    # 1. const_M weight sweep (with L1)
    for cw in [0.1, 0.5, 1.0, 5.0, 10.0]:
        weights = dict(base_cfg.loss_weights)
        weights["const_M"] = cw
        cfg = BeamConfig(**{**base_cfg.__dict__, "loss_weights": weights})
        concrete = SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, tension_model="stiffening", alpha_ts=200)
        steel = BilinearSteel(fy=fy_embedded, Es=base_cfg.Es, b=base_cfg.steel_b)
        tests.append((f"const_M={cw}", cfg, concrete, steel))

    # 2. Learning rate sweep
    for lr in [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]:
        cfg = BeamConfig(**{**base_cfg.__dict__, "learning_rate": lr})
        concrete = SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, tension_model="stiffening", alpha_ts=200)
        steel = BilinearSteel(fy=fy_embedded, Es=base_cfg.Es, b=base_cfg.steel_b)
        tests.append((f"lr={lr:.0e}", cfg, concrete, steel))

    # 3. alpha_ts sweep
    for alpha in [50, 100, 200, 500, 1000]:
        cfg = BeamConfig(**{**base_cfg.__dict__})
        concrete = SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, tension_model="stiffening", alpha_ts=alpha)
        steel = BilinearSteel(fy=fy_embedded, Es=base_cfg.Es, b=base_cfg.steel_b)
        tests.append((f"alpha={alpha}", cfg, concrete, steel))

    results = []
    for i, (label, cfg, concrete, steel) in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {label}")
        r = run_one(cfg, concrete, steel, device, label)
        print(f"  NRMSE_w={r['nrmse_w']:.2f}%, w_mid={r['w_mid']:.2f}/{r['w_ref']:.2f}, RMSE_N={r['rmse_N']:.0f}")
        results.append(r)

    # Summary
    print(f"\n{'Label':<20} {'NRMSE_w%':>10} {'w_mid':>8} {'w_ref':>8} {'RMSE_N':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<20} {r['nrmse_w']:>10.2f} {r['w_mid']:>8.2f} {r['w_ref']:>8.2f} {r['rmse_N']:>10.0f}")

    out_dir = "outputs/sweep_nonlinear"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(f"{'Label':<20} {'NRMSE_w%':>10} {'w_mid':>8} {'w_ref':>8} {'RMSE_N':>10}\n")
        for r in results:
            f.write(f"{r['label']:<20} {r['nrmse_w']:>10.2f} {r['w_mid']:>8.2f} {r['w_ref']:>8.2f} {r['rmse_N']:>10.0f}\n")
    print(f"\nSaved to {out_dir}/results.txt")


if __name__ == "__main__":
    main()
