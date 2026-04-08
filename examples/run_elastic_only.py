#!/usr/bin/env python3
"""Standalone elastic simply supported beam — debug and validate.

Sign convention:
    w positive downward (same direction as load q)
    M positive = sagging (tension on bottom)
    kappa = -w''  (positive when beam sags)
    N positive = tension

Usage:
    cd pinn
    python -m examples.run_elastic_only
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

from configs.simply_supported_elastic import get_config as elastic_config
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer


def make_run_dir(base="outputs"):
    os.makedirs(base, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(base, "run_[0-9][0-9][0-9]")))
    next_num = int(os.path.basename(existing[-1]).split("_")[1]) + 1 if existing else 1
    run_dir = os.path.join(base, f"run_{next_num:03d}")
    os.makedirs(run_dir)
    return run_dir


def compute_section_stiffness(rc):
    EA, ES, EI = 0.0, 0.0, 0.0
    for f in rc.section.fibers.fibers:
        E = f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec
        EA += E * f.area
        ES += E * f.area * f.y
        EI += E * f.area * f.y ** 2
    return EA, ES, EI


def analytical_solution(L, q, EI, x):
    M = q / 2.0 * x * (L - x)
    w = q / (24.0 * EI) * x * (L**3 - 2*L*x**2 + x**3)
    return w, M


def predict_all(pinn, xi, scales, device):
    """Predict network outputs, derivatives, and fiber section outputs.

    Returns dict with dimensional numpy arrays.
    Derivatives are converted to dimensional using chain rule:
        dw/dx = (w_ref / L) * dw_bar/dxi
        dM/dx = (M_ref / L) * dM_bar/dxi
        d2M/dx2 = (M_ref / L^2) * d2M_bar/dxi2
    """
    xi_dev = xi.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)

    w_bar = fields["w_bar"]
    eps0_bar = fields["eps0_bar"]
    M_bar_net = fields["M_bar"]

    # w derivatives: rotation and curvature
    dw_dxi = _grad(w_bar, xi_dev)
    d2w_dxi2 = _grad(dw_dxi, xi_dev)
    kappa_bar = -d2w_dxi2

    # M derivatives: shear and distributed load
    dM_dxi = _grad(M_bar_net, xi_dev)
    d2M_dxi2 = _grad(dM_dxi, xi_dev)

    # Section response (N comes from section, not network)
    eps0_dim = eps0_bar * scales.eps_ref
    kappa_dim = kappa_bar * scales.kap_ref
    N_sec_dim, M_sec_dim = pinn._section_response(eps0_dim, kappa_dim, device)

    L = scales.L_ref
    x = xi.numpy().flatten() * L

    with torch.no_grad():
        result = {
            "x": x,
            "w": w_bar.cpu().numpy().flatten() * scales.w_ref,
            "eps0": eps0_bar.cpu().numpy().flatten() * scales.eps_ref,
            "kappa": kappa_bar.cpu().numpy().flatten() * scales.kap_ref,
            "M_net": M_bar_net.cpu().numpy().flatten() * scales.M_ref,
            "M_sec": M_sec_dim.cpu().numpy().flatten(),
            "N_sec": N_sec_dim.cpu().numpy().flatten(),
            "theta": dw_dxi.cpu().numpy().flatten() * (scales.w_ref / L),
            "V_net": dM_dxi.cpu().numpy().flatten() * (scales.M_ref / L),
            "q_net": -d2M_dxi2.cpu().numpy().flatten() * (scales.M_ref / L**2),
        }
    return result


def main():
    RUN_DIR = make_run_dir()
    print(f">>> Output: {os.path.abspath(RUN_DIR)}")

    cfg = elastic_config()
    device = get_device()
    set_seed(cfg.seed)

    # ── Build section ──
    concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2,
                              Gf=cfg.Gf, h=cfg.concrete_h)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    print(f"  EA={EA:.3e}, ES={ES:.3e}, EI={EI:.3e}")

    L, q = cfg.beam_length, cfg.q
    scales = NonDimScales(L=L, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)

    # ── Analytical ──
    x_ref = np.linspace(0, L, 200)
    w_ref, M_ref = analytical_solution(L, q, EI, x_ref)
    mid_r = len(x_ref) // 2
    print(f"  Analytical midspan: w={w_ref[mid_r]:.4f} mm, M={M_ref[mid_r]:.0f} N.mm")

    # ── PINN setup ──
    fibers = rc.section.fibers
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    # ── Normalization coefficients ──
    nc = scales.norm_coeffs(q, EI)
    print(f"  Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, "
          f"eps0={nc['eps0']:.3e}, N={nc['N']:.3e}")

    field_nets = FieldNetworks(
        hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        norm_coeffs=nc,
        use_fourier=cfg.use_fourier, n_frequencies=cfg.n_frequencies,
        fourier_sigma=cfg.fourier_sigma,
    ).to(device)
    pinn = PINNBeamModel(
        field_nets=field_nets, section=rc.section, scales=scales,
        loss_weights=cfg.loss_weights, elastic=True,
        fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
        N_applied=cfg.N_applied, norm_coeffs=nc,
    )

    xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
    xi_bc = boundary_points(0.0, 1.0, device=device)
    q_bar = scales.to_nondim_q(q)

    optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    trainer = Trainer(pinn, optimizer, scheduler, log_dir=RUN_DIR,
                      use_ntk=False,
                      resample_every=500)

    print(f"\n  Training {cfg.n_epochs} epochs ...")
    logger = trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs)

    # ── Predict ──
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    res = predict_all(pinn, xi_plot, scales, device)
    x = res["x"]
    mid_p = len(x) // 2

    print(f"\n  PINN midspan:     w={res['w'][mid_p]:.4f}, M_net={res['M_net'][mid_p]:.0f}, M_sec={res['M_sec'][mid_p]:.0f}")
    print(f"                    N_sec={res['N_sec'][mid_p]:.0f}, eps0={res['eps0'][mid_p]:.6f}")
    N_app = cfg.N_applied
    eps0_mid_ref = (N_app + ES * M_ref[mid_r] / EI) / EA
    print(f"  Analytical:       w={w_ref[mid_r]:.4f}, M={M_ref[mid_r]:.0f}, eps0={eps0_mid_ref:.6f}, N={N_app:.0f}")

    # ── Analytical derivatives ──
    kappa_ref = M_ref / EI
    # eps0 = (N_applied + ES*kappa) / EA, for asymmetric section with axial load
    eps0_ref = (N_app + ES * kappa_ref) / EA
    # theta = dw/dx = q/(24EI) * (L^3 - 6Lx^2 + 4x^3)
    theta_ref = q / (24.0 * EI) * (L**3 - 6*L*x_ref**2 + 4*x_ref**3)
    # V = dM/dx = q*(L/2 - x)
    V_ref = q * (L / 2.0 - x_ref)
    # q_check = -d2M/dx2 = q (constant)
    q_ref_arr = np.full_like(x_ref, q)

    # ── Plot: 3x3 ──
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: displacement, rotation (w'), curvature (w'')
    ax = axes[0, 0]
    ax.plot(x, res["w"], "b-", lw=2, label="PINN")
    ax.plot(x_ref, w_ref, "r--", lw=1.5, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm) [+down]")
    ax.set_title("Displacement w"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, res["theta"], "b-", lw=2, label="PINN dw/dx")
    ax.plot(x_ref, theta_ref, "r--", lw=1.5, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("theta (rad)")
    ax.set_title("Rotation w'"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(x, res["kappa"], "b-", lw=2, label="PINN kappa")
    ax.plot(x_ref, kappa_ref, "r--", lw=1.5, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("kappa (1/mm)")
    ax.set_title("Curvature -w''"); ax.legend(); ax.grid(True, alpha=0.3)

    # Row 2: moment, shear (M'), load (M'')
    ax = axes[1, 0]
    ax.plot(x, res["M_net"] / 1e6, "b-", lw=2, label="M_net (network)")
    ax.plot(x, res["M_sec"] / 1e6, "g-.", lw=1.5, label="M_sec (fiber)")
    ax.plot(x_ref, M_ref / 1e6, "r--", lw=1.5, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("M (kN.m)")
    ax.set_title("Bending moment M"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, res["V_net"] / 1e3, "b-", lw=2, label="PINN dM/dx")
    ax.plot(x_ref, V_ref / 1e3, "r--", lw=1.5, label="Analytical")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("V (kN)")
    ax.set_title("Shear force M'"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(x, res["q_net"], "b-", lw=2, label="PINN -M''")
    ax.plot(x_ref, q_ref_arr, "r--", lw=1.5, label=f"Analytical q={q}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("q (N/mm)")
    ax.set_title("Distributed load -M''"); ax.legend(); ax.grid(True, alpha=0.3)

    # Row 3: eps0, N, loss
    ax = axes[2, 0]
    ax.plot(x, res["eps0"], "b-", lw=2, label="PINN eps0")
    ax.plot(x_ref, eps0_ref, "r--", lw=1.5, label="Analytical (ES/EA)*kappa")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("eps0")
    ax.set_title("Centroidal strain"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(x, res["N_sec"] / 1e3, "b-", lw=2, label="N_sec (fiber)")
    ax.axhline(N_app / 1e3, color="r", ls="--", lw=1.5, label=f"Ref ({N_app/1e3:.1f} kN)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("N (kN)")
    ax.set_title("Axial force N"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.semilogy(logger.loss_history, "k-", lw=1, label="Total")
    for name, hist in logger.component_history.items():
        if name != "total":
            ax.semilogy(hist, lw=0.8, alpha=0.7, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss history"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Elastic SS Beam: L={L}mm, q={q}N/mm, EI={EI:.3e}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "elastic_fields.png"), dpi=150)
    plt.close(fig)

    # ── NTK weight history ──
    if logger.ntk_weight_history:
        fig_ntk, ax_ntk = plt.subplots(figsize=(10, 5))
        for name, hist in logger.ntk_weight_history.items():
            ax_ntk.plot(hist, lw=1, label=name)
        ax_ntk.set_xlabel("Epoch"); ax_ntk.set_ylabel("NTK Weight")
        ax_ntk.set_title("NTK Adaptive Weights"); ax_ntk.legend(fontsize=8)
        ax_ntk.grid(True, alpha=0.3); ax_ntk.set_yscale("log")
        fig_ntk.tight_layout()
        fig_ntk.savefig(os.path.join(RUN_DIR, "ntk_weights.png"), dpi=150)
        plt.close(fig_ntk)

    # ── Error summary (pointwise MSE) ──
    # Analytical fields on PINN grid (200 points)
    w_ana, M_ana = analytical_solution(L, q, EI, x)
    eps0_ana = (N_app + ES * M_ana / EI) / EA
    N_ana = np.full_like(x, N_app)

    mse_w = np.mean((res["w"] - w_ana) ** 2)
    mse_M = np.mean((res["M_net"] - M_ana) ** 2)
    mse_N = np.mean((res["N_sec"] - N_ana) ** 2)
    mse_eps0 = np.mean((res["eps0"] - eps0_ana) ** 2)

    # Relative MSE (normalized by variance of analytical solution)
    rmse_w = np.sqrt(mse_w) / (np.max(w_ana) - np.min(w_ana)) * 100
    rmse_M = np.sqrt(mse_M) / (np.max(M_ana) - np.min(M_ana)) * 100

    w_err = abs(res["w"][mid_p] - w_ref[mid_r]) / abs(w_ref[mid_r]) * 100
    M_err = abs(res["M_net"][mid_p] - M_ref[mid_r]) / abs(M_ref[mid_r]) * 100

    print(f"\n  ── Error Summary ──")
    print(f"  w midspan:   PINN={res['w'][mid_p]:.4f}, Ref={w_ref[mid_r]:.4f}, Err={w_err:.2f}%")
    print(f"  M midspan:   net={res['M_net'][mid_p]:.0f}, sec={res['M_sec'][mid_p]:.0f}, Ref={M_ref[mid_r]:.0f}, Err={M_err:.2f}%")
    print(f"  max|N_sec|:  {np.max(np.abs(res['N_sec'])):.0f} N")
    print(f"  max|eps0|:   {np.max(np.abs(res['eps0'])):.6f}")
    print(f"\n  ── Pointwise MSE ──")
    print(f"  MSE_w:    {mse_w:.4e}  (NRMSE={rmse_w:.2f}%)")
    print(f"  MSE_M:    {mse_M:.4e}  (NRMSE={rmse_M:.2f}%)")
    print(f"  MSE_N:    {mse_N:.4e}  (RMSE={np.sqrt(mse_N):.0f} N)")
    print(f"  MSE_eps0: {mse_eps0:.4e}  (RMSE={np.sqrt(mse_eps0)*1e6:.2f} με)")

    # ── Save run config & results summary ──
    summary = f"""Run: {os.path.basename(RUN_DIR)}
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

=== Configuration ===
Beam: L={L}mm, b={cfg.section_width}mm, h={cfg.section_height}mm
Load: q={q}N/mm, N_applied={cfg.N_applied}N
Material: Elastic (Ec={cfg.Ec}MPa)
Rebar: {cfg.rebar_layout}

=== PINN Architecture ===
Networks: 3 (w, eps0, M), N from fiber section
Hidden: {cfg.hidden_dims}, activation={cfg.activation}
Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, eps0={nc['eps0']:.3e}, N={nc['N']:.3e}

=== Training ===
Epochs: {cfg.n_epochs}, lr={cfg.learning_rate}
Collocation: {cfg.n_collocation}, resample_every=500
NTK: off
Loss weights: {cfg.loss_weights}

=== Results (vs analytical) ===
w midspan:  PINN={res['w'][mid_p]:.4f}, Ref={w_ref[mid_r]:.4f}, Err={w_err:.2f}%
M midspan:  PINN={res['M_net'][mid_p]:.0f}, Ref={M_ref[mid_r]:.0f}, Err={M_err:.2f}%

MSE_w:    {mse_w:.4e}  (NRMSE={rmse_w:.2f}%)
MSE_M:    {mse_M:.4e}  (NRMSE={rmse_M:.2f}%)
MSE_N:    {mse_N:.4e}  (RMSE={np.sqrt(mse_N):.0f} N)
MSE_eps0: {mse_eps0:.4e}  (RMSE={np.sqrt(mse_eps0)*1e6:.2f} με)
max|N_sec|: {np.max(np.abs(res['N_sec'])):.0f} N

=== Notes ===
"""
    with open(os.path.join(RUN_DIR, "run_summary.txt"), "w") as f:
        f.write(summary)

    print(f"\n  Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
