#!/usr/bin/env python3
"""Sweep network architecture (width × depth) for nonlinear PINN.

Usage:
    cd pinn
    python -m examples.sweep_architecture
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib; matplotlib.use("Agg")

from configs.base_config import BeamConfig
from materials import SmoothConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer
from solvers.section_analysis import solve_beam_nonlinear
from scipy.interpolate import interp1d


def compute_section_stiffness(rc):
    EA, ES, EI = 0.0, 0.0, 0.0
    for f in rc.section.fibers.fibers:
        E = f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec
        EA += E * f.area; ES += E * f.area * f.y; EI += E * f.area * f.y ** 2
    return EA, ES, EI


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_one(label, hidden_dims, cfg, concrete, steel, device):
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

    field_nets = FieldNetworks(hidden_dims=hidden_dims, norm_coeffs=nc).to(device)
    n_params = count_params(field_nets)
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
    trainer = Trainer(pinn, optimizer, None, use_ntk=False, resample_every=500,
                      warmup_keys=["equil_N", "N_sec_bc"],
                      warmup_epochs=cfg.n_epochs // 2)
    trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs, print_every=cfg.n_epochs + 1)

    # Evaluate
    M_mid = q * L**2 / 8
    ref = solve_beam_nonlinear(rc.section, L, q, n_pts=200, kappa_max=M_mid/EI*5)
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    xi_dev = xi_plot.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)
    dw = _grad(fields["w_bar"], xi_dev); d2w = _grad(dw, xi_dev)
    N_sec, _ = pinn._section_response(
        fields["eps0_bar"] * scales.eps_ref, -d2w * scales.kap_ref, device)
    x = xi_plot.numpy().flatten() * L
    with torch.no_grad():
        w = fields["w_bar"].cpu().numpy().flatten() * scales.w_ref
        N = N_sec.cpu().numpy().flatten()

    ref_w = interp1d(ref["x"], ref["w"], kind="cubic")(x)
    ref_N = interp1d(ref["x"], ref["N"], kind="cubic")(x)
    nrmse_w = np.sqrt(np.mean((w - ref_w)**2)) / np.ptp(ref_w) * 100
    rmse_N = np.sqrt(np.mean((N - ref_N)**2))

    return {"label": label, "nrmse_w": nrmse_w, "rmse_N": rmse_N,
            "w_mid": w[100], "w_ref": ref_w[100], "n_params": n_params,
            "hidden": hidden_dims}


def main():
    device = get_device()
    cfg = BeamConfig(
        mode="forward", elastic=False, n_epochs=10000, learning_rate=1e-3,
        n_collocation=200, activation="tanh", q=40.0, N_applied=0.0,
    )
    concrete = SmoothConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co,
                              tension_model="stiffening", alpha_ts=200)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)

    tests = [
        # Width sweep (3 layers)
        ("[16]x3",  [16, 16, 16]),
        ("[32]x3",  [32, 32, 32]),
        ("[64]x3",  [64, 64, 64]),
        ("[128]x3", [128, 128, 128]),
        # Depth sweep (width=32)
        ("[32]x2",  [32, 32]),
        ("[32]x3",  [32, 32, 32]),
        ("[32]x4",  [32, 32, 32, 32]),
        ("[32]x5",  [32, 32, 32, 32, 32]),
        # Depth sweep (width=64)
        ("[64]x2",  [64, 64]),
        ("[64]x4",  [64, 64, 64, 64]),
        # Wide-shallow vs narrow-deep (similar param count)
        ("[128]x2", [128, 128]),
        ("[16]x5",  [16, 16, 16, 16, 16]),
    ]

    results = []
    for i, (label, hdims) in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] {label}")
        r = run_one(label, hdims, cfg, concrete, steel, device)
        print(f"  params={r['n_params']}, NRMSE_w={r['nrmse_w']:.2f}%, "
              f"w={r['w_mid']:.2f}/{r['w_ref']:.2f}, N={r['rmse_N']:.0f}")
        results.append(r)

    # Summary
    print(f"\n{'Label':<12} {'Params':>8} {'NRMSE_w%':>10} {'w_mid':>8} {'w_ref':>8} {'RMSE_N':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["nrmse_w"]):
        print(f"{r['label']:<12} {r['n_params']:>8} {r['nrmse_w']:>10.2f} "
              f"{r['w_mid']:>8.2f} {r['w_ref']:>8.2f} {r['rmse_N']:>10.0f}")

    out_dir = "outputs/sweep_arch"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(f"{'Label':<12} {'Params':>8} {'NRMSE_w%':>10} {'w_mid':>8} {'w_ref':>8} {'RMSE_N':>10}\n")
        for r in sorted(results, key=lambda x: x["nrmse_w"]):
            f.write(f"{r['label']:<12} {r['n_params']:>8} {r['nrmse_w']:>10.2f} "
                    f"{r['w_mid']:>8.2f} {r['w_ref']:>8.2f} {r['rmse_N']:>10.0f}\n")
    print(f"\nSaved to {out_dir}/results.txt")


if __name__ == "__main__":
    main()
