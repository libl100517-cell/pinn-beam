#!/usr/bin/env python3
"""Three-span continuous beam — domain decomposition PINN.

Each span uses standard PINNBeamModel.forward(), coupled at internal
supports by moment and rotation continuity.

     span 0        span 1        span 2
  |___________|___________|___________|
  A     q     B     q     C     q     D
  w=0,M=0    w=0         w=0        w=0,M=0

Usage:
    cd pinn
    python -m examples.run_continuous_beam
"""

import os, sys, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.base_config import BeamConfig
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from physics import NonDimScales
from models import FieldNetworks, PINNBeamModel
from models.pinn_beam import _grad
from utils import get_device, set_seed, uniform_collocation, boundary_points


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
        EA += E * f.area; ES += E * f.area * f.y; EI += E * f.area * f.y ** 2
    return EA, ES, EI


def main():
    RUN_DIR = make_run_dir()
    print(f">>> Output: {os.path.abspath(RUN_DIR)}")

    cfg = BeamConfig(
        n_epochs=20000, learning_rate=1e-3,
        n_collocation=100, activation="tanh",
        q=20.0, N_applied=0.0, elastic=True,
        hidden_dims=[32, 32, 32, 32, 32],
    )
    device = get_device()
    set_seed(cfg.seed)

    n_spans = 3
    L_span = cfg.beam_length
    L_total = n_spans * L_span
    q = cfg.q

    print(f"  {n_spans}-span continuous beam, each L={L_span}mm, total={L_total}mm, q={q}N/mm")

    # Build section (elastic)
    concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2,
                              Gf=cfg.Gf, h=cfg.concrete_h)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
    rc = RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )
    EA, ES, EI = compute_section_stiffness(rc)
    scales = NonDimScales(L=L_span, E_ref=cfg.Ec, A_ref=rc.gross_area, I_ref=rc.gross_inertia)
    nc = scales.norm_coeffs(q, EI)
    print(f"  EA={EA:.3e}, EI={EI:.3e}")

    fibers = rc.section.fibers
    fibers_y = fibers.positions()
    fibers_A = fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]

    # Loss weights: internal spans have M_net_bc=0 (M free at internal supports)
    weights_end = dict(cfg.loss_weights)  # end spans: M=0 at outer end
    weights_mid = dict(cfg.loss_weights)
    weights_mid["M_net_bc"] = 0.0  # middle span: M free at both ends

    # For end spans, we need M=0 only at the outer end.
    # PINNBeamModel enforces M=0 at BOTH ends via M_net_bc.
    # We'll use full weights for all spans, then add a counter-term
    # to free M at internal support ends.

    # Create one PINN per span — M_net_bc=0 for ALL spans
    # M=0 at beam ends will be added manually below
    span_nets = []
    span_pinns = []
    all_params = []
    span_w = dict(cfg.loss_weights)
    span_w["M_net_bc"] = 0.0  # disable built-in M BC, handle manually

    for i in range(n_spans):
        nets = FieldNetworks(
            hidden_dims=cfg.hidden_dims, activation=cfg.activation,
            norm_coeffs=nc,
        ).to(device)
        pinn = PINNBeamModel(
            field_nets=nets, section=rc.section, scales=scales,
            loss_weights=span_w, elastic=cfg.elastic,
            fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
            N_applied=cfg.N_applied, norm_coeffs=nc,
        )
        span_nets.append(nets)
        span_pinns.append(pinn)
        all_params.extend(nets.parameters())

    xi_cols = [uniform_collocation(cfg.n_collocation, 0.0, 1.0, device=device)
               for _ in range(n_spans)]
    xi_bcs = [boundary_points(0.0, 1.0, device=device) for _ in range(n_spans)]
    q_bar = scales.to_nondim_q(q)

    w_compat = 1e5  # compatibility weight

    optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate)

    print(f"\n  Training {cfg.n_epochs} epochs ...")
    loss_history = []

    for epoch in range(1, cfg.n_epochs + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        span_losses = []

        # Each span: standard forward (M_net_bc disabled)
        for i in range(n_spans):
            loss_i, _, _, _ = span_pinns[i].forward(xi_cols[i], xi_bcs[i], q_bar)
            total_loss = total_loss + loss_i
            span_losses.append(loss_i.item())

        # M=0 only at beam ends (A and D)
        w_Mbc = cfg.loss_weights.get("M_net_bc", 100.0)
        bc_A = span_nets[0](xi_bcs[0])
        bc_D = span_nets[-1](xi_bcs[-1])
        total_loss = total_loss + w_Mbc * (bc_A["M_bar"][0:1] ** 2).mean()  # left end
        total_loss = total_loss + w_Mbc * (bc_D["M_bar"][1:2] ** 2).mean()  # right end

        # Compatibility at internal supports
        compat_loss = torch.tensor(0.0, device=device)
        for j in range(n_spans - 1):
            bc_xi_L = xi_bcs[j].detach().requires_grad_(True)
            bc_xi_R = xi_bcs[j + 1].detach().requires_grad_(True)
            f_L = span_nets[j](bc_xi_L)
            f_R = span_nets[j + 1](bc_xi_R)

            # Moment continuity: M_right(span j) = M_left(span j+1)
            dM = f_L["M_bar"][1:2] - f_R["M_bar"][0:1]
            compat_loss = compat_loss + (dM ** 2).mean()

            # Rotation continuity: θ_right(span j) = θ_left(span j+1)
            dw_L = _grad(f_L["w_bar"], bc_xi_L)
            dw_R = _grad(f_R["w_bar"], bc_xi_R)
            dtheta = dw_L[1:2] - dw_R[0:1]
            compat_loss = compat_loss + (dtheta ** 2).mean()

        total_loss = total_loss + w_compat * compat_loss

        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

        if epoch % 2000 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>6d} | total={total_loss.item():.4e} | "
                  f"compat={compat_loss.item():.4e}")

    # ── Predict ──
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    x_all, w_all, M_all, kappa_all = [], [], [], []

    for i in range(n_spans):
        xi_dev = xi_plot.to(device).requires_grad_(True)
        fields = span_nets[i](xi_dev)
        dw = _grad(fields["w_bar"], xi_dev)
        d2w = _grad(dw, xi_dev)
        with torch.no_grad():
            x_all.append(xi_plot.numpy().flatten() * L_span + i * L_span)
            w_all.append(fields["w_bar"].cpu().numpy().flatten() * scales.w_ref)
            M_all.append(fields["M_bar"].cpu().numpy().flatten() * scales.M_ref)
            kappa_all.append((-d2w).cpu().numpy().flatten() * scales.kap_ref)

    x_all = np.concatenate(x_all)
    w_all = np.concatenate(w_all)
    M_all = np.concatenate(M_all)
    kappa_all = np.concatenate(kappa_all)

    # ── Analytical (elastic 3-span equal, uniform load) ──
    M_sup = -q * L_span**2 / 10  # internal support moment
    x_ref = np.linspace(0, L_total, 600)
    M_ref = np.zeros_like(x_ref)
    for idx, xi in enumerate(x_ref):
        si = min(int(xi / L_span), n_spans - 1)
        xl = xi - si * L_span
        Ml = 0.0 if si == 0 else M_sup
        Mr = 0.0 if si == n_spans - 1 else M_sup
        M_ref[idx] = q/2 * xl * (L_span - xl) + Ml*(1 - xl/L_span) + Mr*(xl/L_span)

    kappa_ref = M_ref / EI
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d
    theta_r = cumulative_trapezoid(kappa_ref, x_ref, initial=0)
    w_raw = cumulative_trapezoid(theta_r, x_ref, initial=0)
    sup_x = [i * L_span for i in range(n_spans + 1)]
    sup_w = [np.interp(sx, x_ref, w_raw) for sx in sup_x]
    w_ref = w_raw - interp1d(sup_x, sup_w, kind="linear")(x_ref)

    # Support moments
    print(f"\n  Analytical support moment: {M_sup/1e6:.2f} kN·m")
    for j in range(n_spans - 1):
        with torch.no_grad():
            M_r = span_nets[j](xi_bcs[j])["M_bar"][1].item() * scales.M_ref
            M_l = span_nets[j+1](xi_bcs[j+1])["M_bar"][0].item() * scales.M_ref
        print(f"  Support {j+1}: left={M_r/1e6:.2f}, right={M_l/1e6:.2f} kN·m")

    # MSE
    w_ref_i = np.interp(x_all, x_ref, w_ref)
    M_ref_i = np.interp(x_all, x_ref, M_ref)
    nrmse_w = np.sqrt(np.mean((w_all - w_ref_i)**2)) / np.ptp(w_ref_i) * 100
    nrmse_M = np.sqrt(np.mean((M_all - M_ref_i)**2)) / np.ptp(M_ref_i) * 100
    print(f"\n  NRMSE_w = {nrmse_w:.2f}%")
    print(f"  NRMSE_M = {nrmse_M:.2f}%")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sup_lines = [i * L_span for i in range(n_spans + 1)]

    ax = axes[0, 0]
    ax.plot(x_all, w_all, "b-", lw=2, label="PINN")
    ax.plot(x_ref, w_ref, "r--", lw=1.5, label="Analytical")
    for sx in sup_lines: ax.axvline(sx, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("w (mm)"); ax.set_title("Displacement"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_all, M_all/1e6, "b-", lw=2, label="PINN")
    ax.plot(x_ref, M_ref/1e6, "r--", lw=1.5, label="Analytical")
    for sx in sup_lines: ax.axvline(sx, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("M (kN·m)"); ax.set_title("Bending moment"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x_all, kappa_all, "b-", lw=2, label="PINN")
    ax.plot(x_ref, kappa_ref, "r--", lw=1.5, label="Analytical")
    for sx in sup_lines: ax.axvline(sx, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("κ (1/mm)"); ax.set_title("Curvature"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(loss_history, "k-", lw=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss history"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"3-span continuous beam: L={L_span}mm×3, q={q}N/mm | "
                 f"NRMSE_w={nrmse_w:.2f}%, NRMSE_M={nrmse_M:.2f}%", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "continuous_beam.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
