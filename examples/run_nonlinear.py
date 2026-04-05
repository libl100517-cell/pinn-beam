#!/usr/bin/env python3
"""Nonlinear (elastoplastic) simply supported beam — PINN vs bisection reference.

Usage:
    cd pinn
    python -m examples.run_nonlinear
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

from configs.simply_supported_elastoplastic import get_config
from materials import ManderConcrete, SmoothConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points
from solvers.trainer import Trainer
from solvers.section_analysis import solve_beam_nonlinear, build_M_kappa_curve


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


def predict_all(pinn, xi, scales, device):
    """Predict all fields including section response."""
    xi_dev = xi.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)

    w_bar = fields["w_bar"]
    eps0_bar = fields["eps0_bar"]
    M_bar_net = fields["M_bar"]

    dw_dxi = _grad(w_bar, xi_dev)
    d2w_dxi2 = _grad(dw_dxi, xi_dev)
    kappa_bar = -d2w_dxi2

    # M derivatives: shear and load
    dM_dxi = _grad(M_bar_net, xi_dev)
    d2M_dxi2 = _grad(dM_dxi, xi_dev)

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
            "theta": dw_dxi.cpu().numpy().flatten() * (scales.w_ref / L),
            "V_net": dM_dxi.cpu().numpy().flatten() * (scales.M_ref / L),
            "q_net": -d2M_dxi2.cpu().numpy().flatten() * (scales.M_ref / L**2),
        }


def main():
    RUN_DIR = make_run_dir()
    print(f">>> Output: {os.path.abspath(RUN_DIR)}")

    cfg = get_config()
    device = get_device()
    set_seed(cfg.seed)

    L, q = cfg.beam_length, cfg.q
    N_app = cfg.N_applied
    print(f"  L={L}mm, q={q}N/mm, N_applied={N_app}N")
    print(f"  Section: {cfg.section_width}x{cfg.section_height}mm")
    print(f"  Concrete: fc={cfg.fc}, Ec={cfg.Ec}, eps_co={cfg.eps_co}, eps_cu={cfg.eps_cu}")
    print(f"  Steel: fy={cfg.fy}, Es={cfg.Es}, b={cfg.steel_b}")

    # ── Build section (tension stiffening: concrete + adjusted steel) ──
    concrete = SmoothConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co,
                              tension_model="stiffening", alpha_ts=200)

    # Steel yield reduction: concrete carries part of the tension via bond
    # Δfy = ft · Ac_eff / As, where Ac_eff = b · h_eff (effective tension area)
    h = cfg.section_height
    d_rebar = abs(cfg.rebar_layout[0][0])  # distance from centroid to rebar
    c_bottom = h / 2 - d_rebar             # clear cover
    h_eff = min(2.5 * c_bottom, h / 2)     # effective tension depth (EC2)
    Ac_eff = cfg.section_width * h_eff
    As = sum(a for _, a in cfg.rebar_layout)
    delta_fy = concrete.ft * Ac_eff / As
    fy_embedded = cfg.fy - delta_fy
    print(f"  Tension stiffening: Ac_eff={Ac_eff:.0f}mm², As={As:.0f}mm²")
    print(f"  Δfy = ft·Ac_eff/As = {concrete.ft:.1f}·{Ac_eff:.0f}/{As:.0f} = {delta_fy:.1f} MPa")
    print(f"  fy_embedded = {cfg.fy:.0f} - {delta_fy:.1f} = {fy_embedded:.1f} MPa")

    steel = BilinearSteel(fy=fy_embedded, Es=cfg.Es, b=cfg.steel_b)
    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    print(f"  EA={EA:.3e}, ES={ES:.3e}, EI={EI:.3e} (initial elastic)")

    # ── Reference solution (bisection) ──
    print("\n  === Reference (bisection) ===")
    M_mid = q * L ** 2 / 8
    kappa_max = M_mid / EI * 5  # generous search range
    ref = solve_beam_nonlinear(rc.section, L, q, n_pts=200, kappa_max=kappa_max)

    # ── M-kappa curve ──
    kap_curve, M_curve = build_M_kappa_curve(rc.section, kappa_max=kappa_max, n_pts=100)

    # ── PINN setup ──
    scales = NonDimScales(L=L, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)
    nc = scales.norm_coeffs(q, EI)
    print(f"  Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, "
          f"eps0={nc['eps0']:.3e}, N={nc['N']:.3e}")

    fibers = rc.section.fibers
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    field_nets = FieldNetworks(
        hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        norm_coeffs=nc,
    ).to(device)
    pinn = PINNBeamModel(
        field_nets=field_nets, section=rc.section, scales=scales,
        loss_weights=cfg.loss_weights, elastic=False,  # nonlinear!
        fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
        N_applied=N_app, norm_coeffs=nc,
    )

    xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
    xi_bc = boundary_points(0.0, 1.0, device=device)
    q_bar = scales.to_nondim_q(q)

    optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    trainer = Trainer(pinn, optimizer, scheduler, log_dir=RUN_DIR,
                      use_ntk=False, resample_every=500)

    print(f"\n  Training {cfg.n_epochs} epochs (nonlinear) ...")
    logger = trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs)

    # ── Predict ──
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    res = predict_all(pinn, xi_plot, scales, device)
    x = res["x"]
    mid_p = len(x) // 2
    mid_r = len(ref["x"]) // 2

    print(f"\n  PINN midspan:  w={res['w'][mid_p]:.4f}, M_net={res['M_net'][mid_p]:.0f}, "
          f"M_sec={res['M_sec'][mid_p]:.0f}, eps0={res['eps0'][mid_p]:.6f}")
    print(f"  Ref midspan:   w={ref['w'][mid_r]:.4f}, M={ref['M'][mid_r]:.0f}, "
          f"eps0={ref['eps0'][mid_r]:.6f}")

    # ── Analytical references for shear and load ──
    V_ref = q * (L / 2.0 - ref["x"])
    q_ref_arr = np.full_like(ref["x"], q)

    # ── Plot: 3x4 ──
    # Row 1: w, rotation, curvature, eps0
    # Row 2: M, shear V, distributed load q, N
    # Row 3: M-κ, loss, concrete, steel
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))

    # (0,0) Displacement
    ax = axes[0, 0]
    ax.plot(x, res["w"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["w"], "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm)")
    ax.set_title("Displacement w"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) Rotation
    ax = axes[0, 1]
    ax.plot(x, res["theta"], "b-", lw=2, label="PINN dw/dx")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("θ (rad)")
    ax.set_title("Rotation w'"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,2) Curvature
    ax = axes[0, 2]
    ax.plot(x, res["kappa"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["kappa"], "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("κ (1/mm)")
    ax.set_title("Curvature -w''"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,3) Centroidal strain
    ax = axes[0, 3]
    ax.plot(x, res["eps0"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["eps0"], "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("ε₀")
    ax.set_title("Centroidal strain"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,0) Bending moment
    ax = axes[1, 0]
    ax.plot(x, res["M_net"] / 1e6, "b-", lw=2, label="M_net")
    ax.plot(x, res["M_sec"] / 1e6, "g-.", lw=1.5, label="M_sec")
    ax.plot(ref["x"], ref["M"] / 1e6, "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("M (kN·m)")
    ax.set_title("Bending moment M"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,1) Shear force
    ax = axes[1, 1]
    ax.plot(x, res["V_net"] / 1e3, "b-", lw=2, label="PINN dM/dx")
    ax.plot(ref["x"], V_ref / 1e3, "r--", lw=1.5, label="Ref q(L/2-x)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("V (kN)")
    ax.set_title("Shear force M'"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,2) Distributed load (equilibrium check: -M'' = q)
    ax = axes[1, 2]
    ax.plot(x, res["q_net"], "b-", lw=2, label="PINN -M''")
    ax.plot(ref["x"], q_ref_arr, "r--", lw=1.5, label=f"q={q}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("q (N/mm)")
    ax.set_title("Equilibrium -M''"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,3) Axial force
    ax = axes[1, 3]
    ax.plot(x, res["N_sec"] / 1e3, "b-", lw=2, label="N_sec")
    ax.plot(ref["x"], ref["N"] / 1e3, "r--", lw=1.5, label="Ref")
    ax.axhline(N_app / 1e3, color="gray", ls=":", lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("N (kN)")
    ax.set_title("Axial force N"); ax.legend(); ax.grid(True, alpha=0.3)

    # (2,0) M-κ relationship
    ax = axes[2, 0]
    ax.plot(kap_curve, M_curve / 1e6, "k-", lw=1.5, label="M-κ curve")
    ax.plot(res["kappa"], res["M_sec"] / 1e6, "b.", ms=3, label="PINN")
    ax.plot(ref["kappa"], ref["M"] / 1e6, "r+", ms=5, label="Ref")
    ax.set_xlabel("κ (1/mm)"); ax.set_ylabel("M (kN·m)")
    ax.set_title("M-κ relationship"); ax.legend(); ax.grid(True, alpha=0.3)

    # (2,1) Loss history
    ax = axes[2, 1]
    ax.semilogy(logger.loss_history, "k-", lw=1, label="Total")
    for name, hist in logger.component_history.items():
        if name != "total":
            ax.semilogy(hist, lw=0.8, alpha=0.7, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss history"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (2,2) Concrete
    ax = axes[2, 2]
    eps_c_range = np.linspace(-0.004, 0.003, 300)
    sig_c = [concrete.stress(torch.tensor([e])).item() for e in eps_c_range]
    ax.plot(eps_c_range * 1e3, np.array(sig_c), "b-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title("Concrete σ-ε"); ax.grid(True, alpha=0.3)

    # (2,3) Steel
    ax = axes[2, 3]
    eps_s_range = np.linspace(-0.01, 0.01, 300)
    sig_s = [steel.stress(torch.tensor([e])).item() for e in eps_s_range]
    ax.plot(eps_s_range * 1e3, np.array(sig_s), "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title(f"Steel σ-ε (fy={steel.fy:.0f})"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Nonlinear SS Beam: L={L}mm, q={q}N/mm, fc={cfg.fc}MPa, fy={cfg.fy}MPa",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "nonlinear_fields.png"), dpi=150)
    plt.close(fig)

    # ── Pointwise MSE ──
    # Interpolate reference to PINN grid
    from scipy.interpolate import interp1d
    ref_w_interp = interp1d(ref["x"], ref["w"], kind="cubic")(x)
    ref_M_interp = interp1d(ref["x"], ref["M"], kind="cubic")(x)
    ref_eps0_interp = interp1d(ref["x"], ref["eps0"], kind="cubic")(x)
    ref_kappa_interp = interp1d(ref["x"], ref["kappa"], kind="cubic")(x)
    ref_N_interp = interp1d(ref["x"], ref["N"], kind="cubic")(x)

    mse_w = np.mean((res["w"] - ref_w_interp) ** 2)
    mse_M = np.mean((res["M_net"] - ref_M_interp) ** 2)
    mse_N = np.mean((res["N_sec"] - ref_N_interp) ** 2)
    mse_eps0 = np.mean((res["eps0"] - ref_eps0_interp) ** 2)

    nrmse_w = np.sqrt(mse_w) / np.ptp(ref_w_interp) * 100
    nrmse_M = np.sqrt(mse_M) / np.ptp(ref_M_interp) * 100

    print(f"\n  ── Pointwise MSE (vs bisection reference) ──")
    print(f"  MSE_w:    {mse_w:.4e}  (NRMSE={nrmse_w:.2f}%)")
    print(f"  MSE_M:    {mse_M:.4e}  (NRMSE={nrmse_M:.2f}%)")
    print(f"  MSE_N:    {mse_N:.4e}  (RMSE={np.sqrt(mse_N):.0f} N)")
    print(f"  MSE_eps0: {mse_eps0:.4e}  (RMSE={np.sqrt(mse_eps0)*1e6:.2f} με)")
    print(f"  max|N_sec|: {np.max(np.abs(res['N_sec'])):.0f} N")

    # ── Save run config & results summary ──
    summary = f"""Run: {os.path.basename(RUN_DIR)}
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

=== Configuration ===
Beam: L={L}mm, b={cfg.section_width}mm, h={cfg.section_height}mm
Load: q={q}N/mm, N_applied={N_app}N
Concrete: SmoothConcrete fc={cfg.fc}MPa, Ec={cfg.Ec}MPa, eps_co={cfg.eps_co}
Steel: BilinearSteel fy={cfg.fy}MPa, Es={cfg.Es}MPa, b={cfg.steel_b}
Rebar: {cfg.rebar_layout}
Fibers: {cfg.n_concrete_fibers} concrete fibers

=== PINN Architecture ===
Networks: 3 (w, eps0, M), N from fiber section
Hidden: {cfg.hidden_dims}, activation={cfg.activation}
Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, eps0={nc['eps0']:.3e}, N={nc['N']:.3e}

=== Training ===
Epochs: {cfg.n_epochs}, lr={cfg.learning_rate}
Collocation: {cfg.n_collocation}, resample_every=500
NTK: off
Loss weights: {cfg.loss_weights}

=== Results (vs bisection reference) ===
w midspan:  PINN={res['w'][mid_p]:.4f}, Ref={ref['w'][mid_r]:.4f}
M midspan:  PINN={res['M_net'][mid_p]:.0f}, Ref={ref['M'][mid_r]:.0f}
eps0 midspan: PINN={res['eps0'][mid_p]:.6f}, Ref={ref['eps0'][mid_r]:.6f}

MSE_w:    {mse_w:.4e}  (NRMSE={nrmse_w:.2f}%)
MSE_M:    {mse_M:.4e}  (NRMSE={nrmse_M:.2f}%)
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
