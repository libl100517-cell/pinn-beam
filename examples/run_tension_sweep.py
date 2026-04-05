#!/usr/bin/env python3
"""Sweep concrete tension parameters to find best nonlinear convergence.

Tests:
  1. ft=0 (no tension)
  2. eps_f=10*eps_cr (slow softening)
  3. eps_f=100*eps_cr (very slow softening)
  4. ft=0.5*default (reduced tension)
  5. default (ft=3.4MPa, eps_f=eps_cr)

Usage:
    cd pinn
    python -m examples.run_tension_sweep
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.base_config import BeamConfig
from materials import SmoothConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer
from solvers.section_analysis import solve_beam_nonlinear, build_M_kappa_curve


def compute_section_stiffness(rc):
    EA, ES, EI = 0.0, 0.0, 0.0
    for f in rc.section.fibers.fibers:
        E = f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec
        EA += E * f.area
        ES += E * f.area * f.y
        EI += E * f.area * f.y ** 2
    return EA, ES, EI


def predict_all(pinn, xi, scales, device):
    xi_dev = xi.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)
    w_bar = fields["w_bar"]
    eps0_bar = fields["eps0_bar"]
    M_bar_net = fields["M_bar"]
    dw = _grad(w_bar, xi_dev)
    d2w = _grad(dw, xi_dev)
    kappa_bar = -d2w
    eps0_dim = eps0_bar * scales.eps_ref
    kappa_dim = kappa_bar * scales.kap_ref
    N_sec_dim, M_sec_dim = pinn._section_response(eps0_dim, kappa_dim, device)
    L = scales.L_ref
    x = xi.numpy().flatten() * L
    with torch.no_grad():
        return {
            "x": x,
            "w": w_bar.cpu().numpy().flatten() * scales.w_ref,
            "eps0": eps0_bar.cpu().numpy().flatten() * scales.eps_ref,
            "kappa": kappa_bar.cpu().numpy().flatten() * scales.kap_ref,
            "M_net": M_bar_net.cpu().numpy().flatten() * scales.M_ref,
            "M_sec": M_sec_dim.cpu().numpy().flatten(),
            "N_sec": N_sec_dim.cpu().numpy().flatten(),
        }


def run_single(label, concrete, steel, cfg, device):
    """Run one config, return results dict."""
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
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    field_nets = FieldNetworks(
        hidden_dims=cfg.hidden_dims, activation=cfg.activation, norm_coeffs=nc,
    ).to(device)
    pinn = PINNBeamModel(
        field_nets=field_nets, section=rc.section, scales=scales,
        loss_weights=cfg.loss_weights, elastic=False,
        fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
        N_applied=cfg.N_applied, norm_coeffs=nc,
    )

    xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
    xi_bc = boundary_points(0.0, 1.0, device=device)
    q_bar = scales.to_nondim_q(q)

    optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    trainer = Trainer(pinn, optimizer, scheduler, use_ntk=False, resample_every=500)

    logger = trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs, print_every=cfg.n_epochs + 1)

    # Reference
    M_mid = q * L**2 / 8
    kappa_max = M_mid / EI * 5
    ref = solve_beam_nonlinear(rc.section, L, q, n_pts=200, kappa_max=kappa_max)
    kap_curve, M_curve = build_M_kappa_curve(rc.section, kappa_max=kappa_max, n_pts=100)

    # Predict
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    res = predict_all(pinn, xi_plot, scales, device)
    x = res["x"]

    # MSE
    from scipy.interpolate import interp1d
    ref_w = interp1d(ref["x"], ref["w"], kind="cubic")(x)
    ref_M = interp1d(ref["x"], ref["M"], kind="cubic")(x)
    ref_eps0 = interp1d(ref["x"], ref["eps0"], kind="cubic")(x)
    ref_N = interp1d(ref["x"], ref["N"], kind="cubic")(x)

    mse_w = np.mean((res["w"] - ref_w) ** 2)
    mse_N = np.mean((res["N_sec"] - ref_N) ** 2)
    nrmse_w = np.sqrt(mse_w) / np.ptp(ref_w) * 100

    return {
        "label": label,
        "res": res, "ref": ref, "logger": logger,
        "kap_curve": kap_curve, "M_curve": M_curve,
        "nrmse_w": nrmse_w,
        "rmse_N": np.sqrt(mse_N),
        "w_mid": res["w"][len(x)//2],
        "w_ref": ref["w"][len(ref["x"])//2],
        "concrete": concrete,
    }


def main():
    out_dir = "outputs/tension_sweep"
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> Output: {os.path.abspath(out_dir)}")

    cfg = BeamConfig(
        mode="forward", elastic=False,
        n_epochs=20000, learning_rate=1e-4,
        n_collocation=200, activation="tanh",
        q=40.0, N_applied=0.0,
    )
    device = get_device()
    fc, Ec, eps_co = cfg.fc, cfg.Ec, cfg.eps_co
    ft_default = 0.62 * fc ** 0.5
    eps_cr = ft_default / Ec

    cases = [
        ("ft=0 (no tension)",
         SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, ft=0.001)),  # near-zero ft
        ("eps_f=10x",
         SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, eps_f=eps_cr * 10)),
        ("eps_f=100x",
         SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, eps_f=eps_cr * 100)),
        ("ft=0.5x",
         SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co, ft=ft_default * 0.5)),
        ("default",
         SmoothConcrete(fc=fc, Ec=Ec, eps_co=eps_co)),
    ]

    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)

    results = []
    for i, (label, concrete) in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cases)}] {label}: ft={concrete.ft:.2f}, eps_f={getattr(concrete, 'eps_f', eps_cr)*1e6:.0f}με")
        r = run_single(label, concrete, steel, cfg, device)
        print(f"  NRMSE_w={r['nrmse_w']:.2f}%, w_mid={r['w_mid']:.2f} (ref={r['w_ref']:.2f}), "
              f"RMSE_N={r['rmse_N']:.0f}N")
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Case':<25} {'NRMSE_w%':>10} {'w_mid':>8} {'w_ref':>8} {'RMSE_N':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<25} {r['nrmse_w']:>10.2f} {r['w_mid']:>8.2f} {r['w_ref']:>8.2f} "
              f"{r['rmse_N']:>10.0f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    # Displacement
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(r["res"]["x"], r["res"]["w"], c, lw=1.5, label=r["label"])
    ax.plot(results[0]["ref"]["x"], results[0]["ref"]["w"], "k--", lw=1, label="Ref (varies)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm)"); ax.set_title("Displacement")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Curvature
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(r["res"]["x"], r["res"]["kappa"], c, lw=1.5, label=r["label"])
    ax.plot(results[0]["ref"]["x"], results[0]["ref"]["kappa"], "k--", lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("kappa"); ax.set_title("Curvature")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # eps0
    ax = axes[0, 2]
    for r, c in zip(results, colors):
        ax.plot(r["res"]["x"], r["res"]["eps0"], c, lw=1.5, label=r["label"])
    ax.plot(results[0]["ref"]["x"], results[0]["ref"]["eps0"], "k--", lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("eps0"); ax.set_title("Centroidal strain")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # M-kappa for each
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        ax.plot(r["kap_curve"], r["M_curve"] / 1e6, c, lw=1, alpha=0.5)
        ax.plot(r["res"]["kappa"], r["res"]["M_sec"] / 1e6, ".", color=c, ms=2)
    ax.set_xlabel("kappa"); ax.set_ylabel("M (kN.m)"); ax.set_title("M-κ curves")
    ax.grid(True, alpha=0.3)

    # Material curves
    ax = axes[1, 1]
    eps_range = np.linspace(-0.004, 0.002, 300)
    for (label, concrete), c in zip(cases, colors):
        sig = [concrete.stress(torch.tensor([e])).item() for e in eps_range]
        ax.plot(eps_range * 1e3, sig, c, lw=1.5, label=label)
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title("Concrete curves"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bar chart
    ax = axes[1, 2]
    labels = [r["label"] for r in results]
    nrmses = [r["nrmse_w"] for r in results]
    bars = ax.bar(range(len(labels)), nrmses, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("NRMSE_w (%)")
    ax.set_title("Convergence comparison")
    for bar, v in zip(bars, nrmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Tension parameter sweep: effect on nonlinear PINN convergence", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tension_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved to {os.path.abspath(out_dir)}/tension_sweep.png")


if __name__ == "__main__":
    main()
