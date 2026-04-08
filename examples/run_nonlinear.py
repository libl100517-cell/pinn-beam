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

    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)
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
        use_fourier=cfg.use_fourier, n_frequencies=cfg.n_frequencies,
        fourier_sigma=cfg.fourier_sigma,
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
    scheduler = None  # 常数学习率，不衰减
    trainer = Trainer(pinn, optimizer, scheduler, log_dir=RUN_DIR,
                      use_ntk=False, resample_every=500,
                      warmup_keys=["equil_N", "N_sec_bc"],
                      warmup_epochs=cfg.n_epochs // 2)

    # ── Snapshot function: save comparison plots during training ──
    snap_dir = os.path.join(RUN_DIR, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)

    def save_snapshot(epoch):
        res_snap = predict_all(pinn, xi_plot, scales, device)
        x_snap = res_snap["x"]

        fig_snap, axes_snap = plt.subplots(2, 3, figsize=(18, 8))

        # w
        ax = axes_snap[0, 0]
        ax.plot(x_snap, res_snap["w"], "b-", lw=2, label="PINN")
        ax.plot(ref["x"], ref["w"], "r--", lw=1.5, label="Ref")
        ax.set_title("w"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # kappa
        ax = axes_snap[0, 1]
        ax.plot(x_snap, res_snap["kappa"], "b-", lw=2, label="PINN")
        ax.plot(ref["x"], ref["kappa"], "r--", lw=1.5, label="Ref")
        ax.set_title("κ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # eps0
        ax = axes_snap[0, 2]
        ax.plot(x_snap, res_snap["eps0"], "b-", lw=2, label="PINN")
        ax.plot(ref["x"], ref["eps0"], "r--", lw=1.5, label="Ref")
        ax.set_title("ε₀"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # M
        ax = axes_snap[1, 0]
        ax.plot(x_snap, res_snap["M_net"] / 1e6, "b-", lw=2, label="M_net")
        ax.plot(x_snap, res_snap["M_sec"] / 1e6, "g-.", lw=1.5, label="M_sec")
        ax.plot(ref["x"], ref["M"] / 1e6, "r--", lw=1.5, label="Ref")
        ax.set_title("M (kN·m)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # N
        ax = axes_snap[1, 1]
        ax.plot(x_snap, res_snap["N_sec"] / 1e3, "b-", lw=2, label="N_sec")
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_title("N (kN)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # M-kappa
        ax = axes_snap[1, 2]
        ax.plot(kap_curve, M_curve / 1e6, "k-", lw=1, alpha=0.5)
        ax.plot(res_snap["kappa"], res_snap["M_sec"] / 1e6, "b.", ms=2)
        ax.plot(ref["kappa"], ref["M"] / 1e6, "r+", ms=4)
        ax.set_title("M-κ"); ax.grid(True, alpha=0.3)

        fig_snap.suptitle(f"Epoch {epoch}", fontsize=12)
        fig_snap.tight_layout()
        fig_snap.savefig(os.path.join(snap_dir, f"epoch_{epoch:05d}.png"), dpi=100)
        plt.close(fig_snap)

    print(f"\n  Training {cfg.n_epochs} epochs (nonlinear) ...")
    logger = trainer.train(xi_col, xi_bc, q_bar, cfg.n_epochs,
                           snapshot_fn=save_snapshot, snapshot_every=1000)

    # ── Predict ──
    res = predict_all(pinn, xi_plot, scales, device)
    x = res["x"]
    mid_p = len(x) // 2
    mid_r = len(ref["x"]) // 2

    print(f"\n  PINN midspan:  w={res['w'][mid_p]:.4f}, M_net={res['M_net'][mid_p]:.0f}, "
          f"M_sec={res['M_sec'][mid_p]:.0f}, eps0={res['eps0'][mid_p]:.6f}")
    print(f"  Ref midspan:   w={ref['w'][mid_r]:.4f}, M={ref['M'][mid_r]:.0f}, "
          f"eps0={ref['eps0'][mid_r]:.6f}")

    # ── Reference derivatives ──
    V_ref = q * (L / 2.0 - ref["x"])
    q_ref_arr = np.full_like(ref["x"], q)
    # Rotation from reference: θ = dw/dx ≈ finite difference
    theta_ref = np.gradient(ref["w"], ref["x"])

    # ── Compute top/bottom strains ──
    h = cfg.section_height
    y_top = h / 2.0    # top fiber (compression for sagging)
    y_bot = -h / 2.0   # bottom fiber (tension for sagging)
    # ε = ε₀ - κ·y
    eps_top_pinn = res["eps0"] - res["kappa"] * y_top
    eps_bot_pinn = res["eps0"] - res["kappa"] * y_bot
    eps_top_ref = ref["eps0"] - ref["kappa"] * y_top
    eps_bot_ref = ref["eps0"] - ref["kappa"] * y_bot

    # ── Plot: 4x4 ──
    # Row 1: w, rotation, curvature, eps0
    # Row 2: M, shear V, equilibrium, N
    # Row 3: M-κ, ε₀-κ, ε_top-κ, ε_bot-κ
    # Row 4: loss, concrete, steel, (blank)
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))

    # (0,0) Displacement
    ax = axes[0, 0]
    ax.plot(x, res["w"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["w"], "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm)")
    ax.set_title("Displacement w"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) Rotation
    ax = axes[0, 1]
    ax.plot(x, res["theta"], "b-", lw=2, label="PINN dw/dx")
    ax.plot(ref["x"], theta_ref, "r--", lw=1.5, label="Ref")
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

    # (2,1) ε₀-κ relationship
    ax = axes[2, 1]
    ax.plot(res["kappa"], res["eps0"] * 1e3, "b.", ms=3, label="PINN ε₀")
    ax.plot(ref["kappa"], ref["eps0"] * 1e3, "r+", ms=5, label="Ref ε₀")
    ax.set_xlabel("κ (1/mm)"); ax.set_ylabel("ε₀ (‰)")
    ax.set_title("ε₀-κ"); ax.legend(); ax.grid(True, alpha=0.3)

    # (2,2) ε_top-κ relationship
    ax = axes[2, 2]
    ax.plot(res["kappa"], eps_top_pinn * 1e3, "b.", ms=3, label="PINN ε_top")
    ax.plot(ref["kappa"], eps_top_ref * 1e3, "r+", ms=5, label="Ref ε_top")
    ax.axhline(-2.0, color="gray", ls=":", lw=1, label="ε_co=-2‰")
    ax.set_xlabel("κ (1/mm)"); ax.set_ylabel("ε_top (‰)")
    ax.set_title(f"ε_top-κ (y={y_top:.0f}mm)"); ax.legend(); ax.grid(True, alpha=0.3)

    # (2,3) ε_bot-κ relationship
    ax = axes[2, 3]
    ax.plot(res["kappa"], eps_bot_pinn * 1e3, "b.", ms=3, label="PINN ε_bot")
    ax.plot(ref["kappa"], eps_bot_ref * 1e3, "r+", ms=5, label="Ref ε_bot")
    eps_y = steel.fy / steel.Es
    ax.axhline(eps_y * 1e3, color="gray", ls=":", lw=1, label=f"ε_y={eps_y*1e3:.2f}‰")
    ax.set_xlabel("κ (1/mm)"); ax.set_ylabel("ε_bot (‰)")
    ax.set_title(f"ε_bot-κ (y={y_bot:.0f}mm)"); ax.legend(); ax.grid(True, alpha=0.3)

    # (3,0) Loss history
    ax = axes[3, 0]
    ax.semilogy(logger.loss_history, "k-", lw=1, label="Total")
    for name, hist in logger.component_history.items():
        if name != "total":
            ax.semilogy(hist, lw=0.8, alpha=0.7, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss history"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (3,1) Concrete
    ax = axes[3, 1]
    eps_c_range = np.linspace(-0.004, 0.003, 300)
    sig_c = [concrete.stress(torch.tensor([e])).item() for e in eps_c_range]
    ax.plot(eps_c_range * 1e3, np.array(sig_c), "b-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title("Concrete σ-ε"); ax.grid(True, alpha=0.3)

    # (3,2) Steel
    ax = axes[3, 2]
    eps_s_range = np.linspace(-0.01, 0.01, 300)
    sig_s = [steel.stress(torch.tensor([e])).item() for e in eps_s_range]
    ax.plot(eps_s_range * 1e3, np.array(sig_s), "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title(f"Steel σ-ε (fy={steel.fy:.0f})"); ax.grid(True, alpha=0.3)

    # (3,3) Learning rate + effective weights
    ax = axes[3, 3]
    ax2 = ax.twinx()
    ax.plot(logger.lr_history, "k-", lw=1, label="lr")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("LR & weights")
    # Plot effective weights on secondary axis
    for name, hist in logger.effective_weight_history.items():
        ax2.semilogy(hist, lw=0.7, alpha=0.7, label=name)
    ax2.set_ylabel("Effective weight")
    ax2.legend(fontsize=5, ncol=2, loc="center right")

    fig.suptitle(f"Nonlinear SS Beam: L={L}mm, q={q}N/mm, fc={cfg.fc}MPa, fy={cfg.fy}MPa",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "nonlinear_fields.png"), dpi=150)
    plt.close(fig)

    # ── Stress contour plot (L × h) ──
    n_y = 50  # vertical resolution
    y_arr = np.linspace(-h/2, h/2, n_y)  # bottom to top
    X_grid, Y_grid = np.meshgrid(x, y_arr)

    # Compute concrete stress at every (x, y): σ = concrete.stress(ε₀ - κ·y)
    stress_grid = np.zeros_like(X_grid)
    for j, yj in enumerate(y_arr):
        strain_j = res["eps0"] - res["kappa"] * yj
        strain_t = torch.tensor(strain_j, dtype=torch.float32)
        with torch.no_grad():
            stress_grid[j, :] = concrete.stress(strain_t).numpy()

    # Reference concrete stress field
    from scipy.interpolate import interp1d
    ref_eps0_on_x = interp1d(ref["x"], ref["eps0"], kind="cubic")(x)
    ref_kappa_on_x = interp1d(ref["x"], ref["kappa"], kind="cubic")(x)
    stress_ref_grid = np.zeros_like(X_grid)
    for j, yj in enumerate(y_arr):
        strain_j = ref_eps0_on_x - ref_kappa_on_x * yj
        strain_t = torch.tensor(strain_j, dtype=torch.float32)
        with torch.no_grad():
            stress_ref_grid[j, :] = concrete.stress(strain_t).numpy()

    ft = concrete.ft
    fc_neg = -cfg.fc
    vmin = min(stress_grid.min(), stress_ref_grid.min(), fc_neg * 1.1)
    vmax = max(stress_grid.max(), stress_ref_grid.max(), ft * 1.5)

    fig_s, axes_s = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    for ax_s, sgrid, title in [
        (axes_s[0], stress_grid, "PINN"),
        (axes_s[1], stress_ref_grid, "Reference (bisection)"),
    ]:
        im = ax_s.pcolormesh(X_grid, Y_grid, sgrid, cmap="RdBu_r",
                             vmin=vmin, vmax=vmax, shading="auto")
        # Zero stress contour (neutral axis)
        cs0 = ax_s.contour(X_grid, Y_grid, sgrid, levels=[0],
                           colors="k", linewidths=2)
        ax_s.clabel(cs0, fmt="0", fontsize=8)
        # Tension limit contour
        if sgrid.max() > ft > 0:
            cs_t = ax_s.contour(X_grid, Y_grid, sgrid, levels=[ft],
                                colors="lime", linewidths=1.5, linestyles="--")
            ax_s.clabel(cs_t, fmt=f"ft={ft:.1f}", fontsize=7)
        # Compression limit contour
        if sgrid.min() < fc_neg:
            cs_c = ax_s.contour(X_grid, Y_grid, sgrid, levels=[fc_neg],
                                colors="cyan", linewidths=1.5, linestyles="--")
            ax_s.clabel(cs_c, fmt=f"fc={fc_neg:.0f}", fontsize=7)
        # Mark rebar positions
        for y_rebar, _ in cfg.rebar_layout:
            ax_s.axhline(y_rebar, color="gray", ls="-", lw=0.5, alpha=0.5)
        ax_s.set_ylabel("y (mm)")
        ax_s.set_title(title)
        ax_s.set_aspect("auto")
        fig_s.colorbar(im, ax=ax_s, label="σ_concrete (MPa)", shrink=0.8)

    axes_s[1].set_xlabel("x (mm)")
    fig_s.suptitle("Concrete stress contour", fontsize=12)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(RUN_DIR, "stress_contour.png"), dpi=150)
    plt.close(fig_s)
    print(f"  Stress contour saved.")

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
