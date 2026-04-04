#!/usr/bin/env python3
"""Full nonlinear beam via curriculum learning.

Strategy: gradually increase load from elastic to full nonlinear,
using previous stage's network weights as initialization.

Stage 1: q_target * 0.2  (elastic range)
Stage 2: q_target * 0.4
Stage 3: q_target * 0.6
Stage 4: q_target * 0.8
Stage 5: q_target * 1.0  (full load, deep into nonlinear)

Usage:
    cd pinn
    python -m examples.run_nonlinear_curriculum
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
    xi_dev = xi.to(device).requires_grad_(True)
    fields = pinn.field_nets(xi_dev)
    w_bar = fields["w_bar"]
    eps0_bar = fields["eps0_bar"]
    M_bar_net = fields["M_bar"]

    dw_dxi = _grad(w_bar, xi_dev)
    d2w_dxi2 = _grad(dw_dxi, xi_dev)
    kappa_bar = -d2w_dxi2

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


def main():
    RUN_DIR = make_run_dir()
    print(f">>> Output: {os.path.abspath(RUN_DIR)}")

    q_target = 40.0
    load_stages = [0.2, 0.4, 0.6, 0.8, 1.0]
    epochs_per_stage = 10000

    cfg = BeamConfig(
        mode="forward",
        elastic=False,
        n_epochs=epochs_per_stage,
        learning_rate=1e-4,
        n_collocation=200,
        activation="tanh",
        q=q_target,
        N_applied=0.0,
    )
    device = get_device()
    set_seed(cfg.seed)

    L = cfg.beam_length
    print(f"  L={L}mm, q_target={q_target}N/mm")
    print(f"  Stages: {[f'{s*q_target:.0f}N/mm' for s in load_stages]}")
    print(f"  Epochs per stage: {epochs_per_stage}")

    # Build section with full nonlinear materials
    concrete = SmoothConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)
    print(f"  Concrete: SmoothConcrete fc={cfg.fc}, Ec={cfg.Ec}, eps_co={cfg.eps_co}")
    print(f"  Steel: BilinearSteel fy={cfg.fy}, Es={cfg.Es}, b={cfg.steel_b}")

    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    print(f"  EA={EA:.3e}, ES={ES:.3e}, EI={EI:.3e}")

    scales = NonDimScales(L=L, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)

    fibers = rc.section.fibers
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    # Build network (shared across stages)
    nc = scales.norm_coeffs(q_target, EI)
    print(f"  Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, eps0={nc['eps0']:.3e}")

    field_nets = FieldNetworks(
        hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        norm_coeffs=nc,
    ).to(device)

    xi_col = uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
    xi_bc = boundary_points(0.0, 1.0, device=device)

    all_loss_history = []
    stage_results = []

    for stage_idx, load_frac in enumerate(load_stages):
        q_stage = q_target * load_frac
        q_bar = scales.to_nondim_q(q_stage)

        print(f"\n  ===== Stage {stage_idx+1}/{len(load_stages)}: "
              f"q={q_stage:.1f} N/mm ({load_frac*100:.0f}%) =====")

        pinn = PINNBeamModel(
            field_nets=field_nets, section=rc.section, scales=scales,
            loss_weights=cfg.loss_weights, elastic=False,
            fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
            N_applied=cfg.N_applied, norm_coeffs=nc,
        )

        # Fresh optimizer each stage (reset momentum)
        optimizer = torch.optim.Adam(field_nets.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_stage)
        trainer = Trainer(pinn, optimizer, scheduler,
                          use_ntk=False, resample_every=500)

        logger = trainer.train(xi_col, xi_bc, q_bar, epochs_per_stage,
                               print_every=epochs_per_stage)
        all_loss_history.extend(logger.loss_history)

        # Evaluate this stage
        xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
        res = predict_all(pinn, xi_plot, scales, device)
        mid_p = len(res["x"]) // 2
        print(f"    w_mid={res['w'][mid_p]:.4f}, M_net={res['M_net'][mid_p]:.0f}, "
              f"eps0={res['eps0'][mid_p]:.6f}, max|N|={np.max(np.abs(res['N_sec'])):.0f}")
        stage_results.append((q_stage, res))

    # Reference solution at full load
    print(f"\n  === Reference (bisection) at q={q_target} ===")
    M_mid = q_target * L**2 / 8
    kappa_max = M_mid / EI * 5
    ref = solve_beam_nonlinear(rc.section, L, q_target, n_pts=200, kappa_max=kappa_max)
    kap_curve, M_curve = build_M_kappa_curve(rc.section, kappa_max=kappa_max, n_pts=100)

    # Final evaluation
    _, res_final = stage_results[-1]
    x = res_final["x"]
    mid_p = len(x) // 2
    mid_r = len(ref["x"]) // 2

    print(f"\n  PINN midspan:  w={res_final['w'][mid_p]:.4f}, M={res_final['M_net'][mid_p]:.0f}")
    print(f"  Ref midspan:   w={ref['w'][mid_r]:.4f}, M={ref['M'][mid_r]:.0f}")

    # Plot 3x3
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    ax = axes[0, 0]
    ax.plot(x, res_final["w"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["w"], "r--", lw=1.5, label="Bisection ref")
    for q_s, res_s in stage_results[:-1]:
        ax.plot(res_s["x"], res_s["w"], "-", lw=0.8, alpha=0.4, label=f"q={q_s:.0f}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm)")
    ax.set_title("Displacement w"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, res_final["eps0"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["eps0"], "r--", lw=1.5, label="Bisection ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("eps0")
    ax.set_title("Centroidal strain"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(x, res_final["kappa"], "b-", lw=2, label="PINN")
    ax.plot(ref["x"], ref["kappa"], "r--", lw=1.5, label="Bisection ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("kappa (1/mm)")
    ax.set_title("Curvature"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, res_final["M_net"] / 1e6, "b-", lw=2, label="M_net")
    ax.plot(x, res_final["M_sec"] / 1e6, "g-.", lw=1.5, label="M_sec")
    ax.plot(ref["x"], ref["M"] / 1e6, "r--", lw=1.5, label="Ref")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("M (kN.m)")
    ax.set_title("Bending moment"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, res_final["N_sec"] / 1e3, "b-", lw=2, label="N_sec")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("N (kN)")
    ax.set_title("Axial force"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(kap_curve, M_curve / 1e6, "k-", lw=1.5, label="M-κ curve")
    ax.plot(res_final["kappa"], res_final["M_sec"] / 1e6, "b.", ms=3, label="PINN")
    ax.plot(ref["kappa"], ref["M"] / 1e6, "r+", ms=5, label="Ref")
    ax.set_xlabel("kappa (1/mm)"); ax.set_ylabel("M (kN.m)")
    ax.set_title("M-κ relationship"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(x, (res_final["M_net"] - res_final["M_sec"]) / 1e3, "b-", lw=1.5)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("ΔM (kN.mm)")
    ax.set_title("Constitutive gap"); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.semilogy(all_loss_history, "k-", lw=0.5)
    for i, frac in enumerate(load_stages):
        ax.axvline(i * epochs_per_stage, color="gray", ls=":", lw=0.8)
        ax.text(i * epochs_per_stage + 100, max(all_loss_history) * 0.5,
                f"{frac*100:.0f}%", fontsize=7)
    ax.set_xlabel("Epoch (total)"); ax.set_ylabel("Loss")
    ax.set_title("Loss history (all stages)"); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    eps_c = np.linspace(-0.004, 0.001, 200)
    sig_c = [concrete.stress(torch.tensor([e])).item() for e in eps_c]
    eps_s = np.linspace(-0.005, 0.005, 200)
    sig_s = [steel.stress(torch.tensor([e])).item() for e in eps_s]
    ax.plot(eps_c * 1e3, sig_c, "b-", lw=1.5, label="Concrete")
    ax.plot(eps_s * 1e3, sig_s, "r-", lw=1.5, label="Steel")
    ax.set_xlabel("strain (‰)"); ax.set_ylabel("stress (MPa)")
    ax.set_title("Material curves"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Curriculum learning: q=0→{q_target}N/mm in {len(load_stages)} stages",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "curriculum_nonlinear.png"), dpi=150)
    plt.close(fig)

    # Pointwise MSE
    from scipy.interpolate import interp1d
    ref_w = interp1d(ref["x"], ref["w"], kind="cubic")(x)
    ref_M = interp1d(ref["x"], ref["M"], kind="cubic")(x)
    ref_eps0 = interp1d(ref["x"], ref["eps0"], kind="cubic")(x)
    ref_N = interp1d(ref["x"], ref["N"], kind="cubic")(x)

    mse_w = np.mean((res_final["w"] - ref_w) ** 2)
    mse_M = np.mean((res_final["M_net"] - ref_M) ** 2)
    mse_N = np.mean((res_final["N_sec"] - ref_N) ** 2)
    mse_eps0 = np.mean((res_final["eps0"] - ref_eps0) ** 2)

    print(f"\n  ── Pointwise MSE (vs bisection at q={q_target}) ──")
    print(f"  MSE_w:    {mse_w:.4e}  (NRMSE={np.sqrt(mse_w)/np.ptp(ref_w)*100:.2f}%)")
    print(f"  MSE_M:    {mse_M:.4e}  (NRMSE={np.sqrt(mse_M)/np.ptp(ref_M)*100:.2f}%)")
    print(f"  MSE_N:    {mse_N:.4e}  (RMSE={np.sqrt(mse_N):.0f} N)")
    print(f"  MSE_eps0: {mse_eps0:.4e}  (RMSE={np.sqrt(mse_eps0)*1e6:.2f} με)")
    print(f"  max|N_sec|: {np.max(np.abs(res_final['N_sec'])):.0f} N")
    print(f"\n  Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
