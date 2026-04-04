#!/usr/bin/env python3
"""Sweep loss weights to find optimal combination.

Tests different equil_N and N_sec_bc weight combinations and reports
pointwise MSE for all fields.

Usage:
    cd pinn
    python -m examples.sweep_weights
"""

import os
import sys
import itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.simply_supported_elastic import get_config as elastic_config
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer
from examples.run_elastic_only import (
    compute_section_stiffness, analytical_solution, predict_all,
)


def run_single(equil_N_w, N_sec_bc_w, cfg, device, EA, ES, EI, scales,
               fibers_y, fibers_A, fibers_is_steel, rc, nc):
    """Run one training with given weights, return MSE dict."""
    set_seed(cfg.seed)

    weights = dict(cfg.loss_weights)
    weights["equil_N"] = equil_N_w
    weights["N_sec_bc"] = N_sec_bc_w

    field_nets = FieldNetworks(
        hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        norm_coeffs=nc,
    ).to(device)
    pinn = PINNBeamModel(
        field_nets=field_nets, section=rc.section, scales=scales,
        loss_weights=weights, elastic=True,
        fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
        N_applied=cfg.N_applied, norm_coeffs=nc,
    )

    L, q = cfg.beam_length, cfg.q
    xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
    xi_bc = boundary_points(0.0, 1.0, device=device)
    q_bar = scales.to_nondim_q(q)

    optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    trainer = Trainer(pinn, optimizer, scheduler,
                      use_ntk=True, ntk_every=100, ntk_alpha=0.1,
                      resample_every=500)

    trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs, print_every=cfg.n_epochs + 1)

    # Evaluate
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    res = predict_all(pinn, xi_plot, scales, device)
    x = res["x"]
    N_app = cfg.N_applied

    w_ana, M_ana = analytical_solution(L, q, EI, x)
    eps0_ana = (N_app + ES * M_ana / EI) / EA
    N_ana = np.full_like(x, N_app)

    return {
        "equil_N": equil_N_w,
        "N_sec_bc": N_sec_bc_w,
        "mse_w": np.mean((res["w"] - w_ana) ** 2),
        "mse_M": np.mean((res["M_net"] - M_ana) ** 2),
        "mse_N": np.mean((res["N_sec"] - N_ana) ** 2),
        "mse_eps0": np.mean((res["eps0"] - eps0_ana) ** 2),
        "rmse_N": np.sqrt(np.mean((res["N_sec"] - N_ana) ** 2)),
        "rmse_eps0_ue": np.sqrt(np.mean((res["eps0"] - eps0_ana) ** 2)) * 1e6,
        "nrmse_w": np.sqrt(np.mean((res["w"] - w_ana) ** 2)) / np.ptp(w_ana) * 100,
        "nrmse_M": np.sqrt(np.mean((res["M_net"] - M_ana) ** 2)) / np.ptp(M_ana) * 100,
    }


def main():
    cfg = elastic_config()
    device = get_device()

    # Build section
    concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2,
                              Gf=cfg.Gf, h=cfg.concrete_h)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    L, q = cfg.beam_length, cfg.q
    scales = NonDimScales(L=L, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)
    nc = scales.norm_coeffs(q, EI)

    fibers = rc.section.fibers
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    # Sweep parameters
    equil_N_values = [1, 10, 100, 1e3, 1e4, 1e5]
    N_sec_bc_values = [10, 100, 1e3, 1e4, 1e5, 1e6]

    results = []
    total = len(equil_N_values) * len(N_sec_bc_values)
    for i, (en, nb) in enumerate(itertools.product(equil_N_values, N_sec_bc_values)):
        print(f"\n[{i+1}/{total}] equil_N={en:.0e}, N_sec_bc={nb:.0e}")
        r = run_single(en, nb, cfg, device, EA, ES, EI, scales,
                       fibers_y, fibers_A, fibers_is_steel, rc, nc)
        print(f"  NRMSE_w={r['nrmse_w']:.2f}%, NRMSE_M={r['nrmse_M']:.2f}%, "
              f"RMSE_N={r['rmse_N']:.0f}N, RMSE_eps0={r['rmse_eps0_ue']:.2f}με")
        results.append(r)

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'equil_N':>10} {'N_sec_bc':>10} | {'NRMSE_w%':>10} {'NRMSE_M%':>10} "
          f"{'RMSE_N(N)':>10} {'RMSE_ε₀(με)':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['equil_N']:>10.0e} {r['N_sec_bc']:>10.0e} | {r['nrmse_w']:>10.2f} "
              f"{r['nrmse_M']:>10.2f} {r['rmse_N']:>10.0f} {r['rmse_eps0_ue']:>12.2f}")

    # Plot heatmaps
    en_vals = sorted(set(r["equil_N"] for r in results))
    nb_vals = sorted(set(r["N_sec_bc"] for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("nrmse_w", "NRMSE_w (%)"),
        ("nrmse_M", "NRMSE_M (%)"),
        ("rmse_N", "RMSE_N (N)"),
        ("rmse_eps0_ue", "RMSE_eps0 (με)"),
    ]
    for ax, (key, title) in zip(axes.flatten(), metrics):
        grid = np.zeros((len(en_vals), len(nb_vals)))
        for r in results:
            i = en_vals.index(r["equil_N"])
            j = nb_vals.index(r["N_sec_bc"])
            grid[i, j] = r[key]
        im = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(len(nb_vals)))
        ax.set_xticklabels([f"{v:.0e}" for v in nb_vals], rotation=45, fontsize=7)
        ax.set_yticks(range(len(en_vals)))
        ax.set_yticklabels([f"{v:.0e}" for v in en_vals], fontsize=7)
        ax.set_xlabel("N_sec_bc"); ax.set_ylabel("equil_N")
        ax.set_title(title)
        for ii in range(len(en_vals)):
            for jj in range(len(nb_vals)):
                ax.text(jj, ii, f"{grid[ii, jj]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Weight Sweep: equil_N vs N_sec_bc", fontsize=14)
    fig.tight_layout()
    out_dir = "outputs/sweep"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "weight_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved to {os.path.abspath(out_dir)}/weight_sweep.png")


if __name__ == "__main__":
    main()
