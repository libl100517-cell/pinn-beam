#!/usr/bin/env python3
"""Three-span continuous beam — domain decomposition PINN.

Each span uses standard PINNBeamModel, coupled at internal supports by:
  1. w = 0 at all supports (built-in BC)
  2. M continuity at internal supports (compat loss)
  3. θ continuity at internal supports (compat loss)
  4. M = 0 only at beam ends A and D

Usage:
    cd pinn
    python -m examples.run_continuous_beam
"""

import os, sys, glob, numpy as np, datetime
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


def predict_span(nets, pinn, xi, scales, device):
    """Predict all fields for one span."""
    xi_dev = xi.to(device).requires_grad_(True)
    fields = nets(xi_dev)
    dw = _grad(fields["w_bar"], xi_dev)
    d2w = _grad(dw, xi_dev)
    kappa_bar = -d2w
    dM = _grad(fields["M_bar"], xi_dev)
    d2M = _grad(dM, xi_dev)
    eps0_dim = fields["eps0_bar"] * scales.eps_ref
    kappa_dim = kappa_bar * scales.kap_ref
    N_sec, M_sec = pinn._section_response(eps0_dim, kappa_dim, device)
    L = scales.L_ref
    with torch.no_grad():
        return {
            "w": fields["w_bar"].cpu().numpy().flatten() * scales.w_ref,
            "eps0": fields["eps0_bar"].cpu().numpy().flatten() * scales.eps_ref,
            "kappa": kappa_bar.cpu().numpy().flatten() * scales.kap_ref,
            "M_net": fields["M_bar"].cpu().numpy().flatten() * scales.M_ref,
            "M_sec": M_sec.cpu().numpy().flatten(),
            "N_sec": N_sec.cpu().numpy().flatten(),
            "theta": dw.cpu().numpy().flatten() * (scales.w_ref / L),
            "V_net": dM.cpu().numpy().flatten() * (scales.M_ref / L),
            "q_net": -d2M.cpu().numpy().flatten() * (scales.M_ref / L**2),
        }


def main():
    RUN_DIR = make_run_dir()
    print(f">>> Output: {os.path.abspath(RUN_DIR)}")

    cfg = BeamConfig(
        n_epochs=20000, learning_rate=1e-3,
        n_collocation=100, activation="tanh",
        q=20.0, N_applied=0.0, elastic=True,
        hidden_dims=[32, 32, 32, 32, 32],
        loss_weights={
            "equil_M": 100.0,
            "equil_N": 1e4,
            "const_M": 10.0,
            "bc": 1000.0,        # stronger w=0 at supports
            "M_net_bc": 100.0,
            "M_sec_bc": 1000.0,  # stronger M_sec=0 at boundaries
            "N_sec_bc": 1e6,
            "data_disp": 1.0,
        },
    )
    device = get_device()
    set_seed(cfg.seed)

    n_spans = 3
    L_span = cfg.beam_length
    L_total = n_spans * L_span
    q = cfg.q

    print(f"  {n_spans}-span continuous beam, L={L_span}mm×{n_spans}, q={q}N/mm")

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
    fibers = rc.section.fibers
    fibers_y, fibers_A = fibers.positions(), fibers.areas()
    fibers_is_steel = [isinstance(f.material, BilinearSteel) for f in fibers.fibers]
    print(f"  EA={EA:.3e}, EI={EI:.3e}")

    # Create PINNs — M_net_bc=0 (manual M=0 at beam ends only)
    span_w = dict(cfg.loss_weights)
    span_w["M_net_bc"] = 0.0

    span_nets, span_pinns, all_params = [], [], []
    for i in range(n_spans):
        nets = FieldNetworks(hidden_dims=cfg.hidden_dims, activation=cfg.activation,
                             norm_coeffs=nc).to(device)
        pinn = PINNBeamModel(
            field_nets=nets, section=rc.section, scales=scales,
            loss_weights=span_w, elastic=cfg.elastic,
            fibers_y=fibers_y, fibers_A=fibers_A, fibers_is_steel=fibers_is_steel,
            N_applied=cfg.N_applied, norm_coeffs=nc,
        )
        span_nets.append(nets); span_pinns.append(pinn)
        all_params.extend(nets.parameters())

    xi_cols = [uniform_collocation(cfg.n_collocation, device=device) for _ in range(n_spans)]
    xi_bcs = [boundary_points(device=device) for _ in range(n_spans)]
    q_bar = scales.to_nondim_q(q)
    w_compat = 1e4
    w_Mbc = cfg.loss_weights.get("M_net_bc", 100.0)

    optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate)
    xi_plot = torch.linspace(0, 1, 200).unsqueeze(1)
    sup_x = [i * L_span for i in range(n_spans + 1)]

    # Analytical reference
    M_sup = -q * L_span**2 / 10
    x_ref = np.linspace(0, L_total, 600)
    M_ref = np.zeros_like(x_ref)
    for idx, xi in enumerate(x_ref):
        si = min(int(xi / L_span), n_spans - 1)
        xl = xi - si * L_span
        Ml = 0.0 if si == 0 else M_sup
        Mr = 0.0 if si == n_spans - 1 else M_sup
        M_ref[idx] = q/2*xl*(L_span-xl) + Ml*(1-xl/L_span) + Mr*(xl/L_span)
    kappa_ref = M_ref / EI
    V_ref = np.gradient(M_ref, x_ref)
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d
    # w'' = -kappa → integrate -kappa twice for w
    theta_raw = cumulative_trapezoid(-kappa_ref, x_ref, initial=0)
    w_raw = cumulative_trapezoid(theta_raw, x_ref, initial=0)
    # Enforce w=0 at all supports
    sw = [np.interp(sx, x_ref, w_raw) for sx in sup_x]
    w_ref = w_raw - interp1d(sup_x, sw, kind="linear")(x_ref)
    theta_ref = np.gradient(w_ref, x_ref)
    eps0_ref = (ES / EA) * kappa_ref

    # Snapshot function
    snap_dir = os.path.join(RUN_DIR, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    def save_snapshot(epoch):
        x_a, w_a, M_a = [], [], []
        for i in range(n_spans):
            r = predict_span(span_nets[i], span_pinns[i], xi_plot, scales, device)
            x_a.append(xi_plot.numpy().flatten() * L_span + i * L_span)
            w_a.append(r["w"]); M_a.append(r["M_net"])
        x_a, w_a, M_a = np.concatenate(x_a), np.concatenate(w_a), np.concatenate(M_a)
        fig_s, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(x_a, w_a, "b-", lw=2); a1.plot(x_ref, w_ref, "r--", lw=1)
        for sx in sup_x: a1.axvline(sx, color="gray", ls=":", lw=0.5)
        a1.set_title(f"w (epoch {epoch})"); a1.grid(True, alpha=0.3)
        a2.plot(x_a, M_a/1e6, "b-", lw=2); a2.plot(x_ref, M_ref/1e6, "r--", lw=1)
        for sx in sup_x: a2.axvline(sx, color="gray", ls=":", lw=0.5)
        a2.set_title(f"M (epoch {epoch})"); a2.grid(True, alpha=0.3)
        fig_s.tight_layout()
        fig_s.savefig(os.path.join(snap_dir, f"epoch_{epoch:05d}.png"), dpi=100)
        plt.close(fig_s)

    # Training
    print(f"\n  Training {cfg.n_epochs} epochs ...")
    loss_history, lr_history, compat_history = [], [], []
    comp_history = {}  # per-component history (summed across spans)
    weight_history = {}  # effective weight history
    best_loss, best_state = float("inf"), {}

    for epoch in range(1, cfg.n_epochs + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # Warmup ramp: 0→1 over first half
        warmup_ramp = min(epoch / (cfg.n_epochs // 2), 1.0)

        # N weights warmup (same as single span)
        n_warmup = {"equil_N": warmup_ramp, "N_sec_bc": warmup_ramp}

        # Accumulate loss components across spans
        epoch_comps = {}
        for i in range(n_spans):
            loss_i, comps_i, _, _ = span_pinns[i].forward(
                xi_cols[i], xi_bcs[i], q_bar, adaptive_weights=n_warmup)
            total_loss = total_loss + loss_i
            for k, v in comps_i.items():
                if k != "total":
                    epoch_comps[k] = epoch_comps.get(k, 0) + v

        # M=0 at beam ends only (warmup)
        bc_A = span_nets[0](xi_bcs[0])
        bc_D = span_nets[-1](xi_bcs[-1])
        M_end_loss = w_Mbc * warmup_ramp * (
            (bc_A["M_bar"][0:1]**2).mean() + (bc_D["M_bar"][1:2]**2).mean())
        total_loss = total_loss + M_end_loss
        epoch_comps["M_end_bc"] = M_end_loss.item()

        # Compatibility (warmup)
        compat_M_loss = torch.tensor(0.0, device=device)
        compat_theta_loss = torch.tensor(0.0, device=device)
        compat_eps0_loss = torch.tensor(0.0, device=device)
        for j in range(n_spans - 1):
            bL = xi_bcs[j].detach().requires_grad_(True)
            bR = xi_bcs[j+1].detach().requires_grad_(True)
            fL, fR = span_nets[j](bL), span_nets[j+1](bR)
            # M continuity
            compat_M_loss = compat_M_loss + (fL["M_bar"][1:2] - fR["M_bar"][0:1])**2
            # θ continuity
            dwL = _grad(fL["w_bar"], bL)
            dwR = _grad(fR["w_bar"], bR)
            compat_theta_loss = compat_theta_loss + (dwL[1:2] - dwR[0:1])**2
            # eps0 continuity (→ N continuity via section response)
            compat_eps0_loss = compat_eps0_loss + (fL["eps0_bar"][1:2] - fR["eps0_bar"][0:1])**2
        compat_loss = compat_M_loss.sum() + compat_theta_loss.sum() + compat_eps0_loss.sum()
        total_loss = total_loss + w_compat * warmup_ramp * compat_loss
        epoch_comps["compat_M"] = compat_M_loss.item()
        epoch_comps["compat_θ"] = compat_theta_loss.item()
        epoch_comps["compat_ε₀"] = compat_eps0_loss.item()

        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        lr_history.append(optimizer.param_groups[0]["lr"])
        compat_history.append(compat_loss.item())

        # Track effective weights (manual × warmup)
        eff_w = {}
        for k, v in span_w.items():
            ramp = warmup_ramp if k in ("equil_N", "N_sec_bc") else 1.0
            eff_w[k] = v * ramp
        eff_w["M_end_bc"] = w_Mbc * warmup_ramp
        eff_w["compat"] = w_compat * warmup_ramp
        for k, v in eff_w.items():
            if k not in weight_history:
                weight_history[k] = []
            weight_history[k].append(v)
        for k, v in epoch_comps.items():
            if k not in comp_history:
                comp_history[k] = []
            comp_history[k].append(v)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {f"span{i}": {k: v.clone() for k, v in span_nets[i].state_dict().items()}
                          for i in range(n_spans)}

        if epoch % 2000 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>6d} | loss={total_loss.item():.4e} | compat={compat_loss.item():.4e}")
        if epoch % 5000 == 0:
            save_snapshot(epoch)

    # Save weights
    wt_dir = os.path.join(RUN_DIR, "weights")
    os.makedirs(wt_dir, exist_ok=True)
    for i in range(n_spans):
        torch.save(span_nets[i].state_dict(), os.path.join(wt_dir, f"span{i}_last.pt"))
    torch.save(best_state, os.path.join(wt_dir, "best_all.pt"))
    print(f"  Saved best (loss={best_loss:.4e}) and last weights.")

    # Predict all spans
    x_all, w_all, M_all, kappa_all, eps0_all = [], [], [], [], []
    theta_all, V_all, q_all, N_all, M_sec_all = [], [], [], [], []
    for i in range(n_spans):
        r = predict_span(span_nets[i], span_pinns[i], xi_plot, scales, device)
        x_span = xi_plot.numpy().flatten() * L_span + i * L_span
        x_all.append(x_span); w_all.append(r["w"]); M_all.append(r["M_net"])
        kappa_all.append(r["kappa"]); eps0_all.append(r["eps0"])
        theta_all.append(r["theta"]); V_all.append(r["V_net"])
        q_all.append(r["q_net"]); N_all.append(r["N_sec"]); M_sec_all.append(r["M_sec"])

    x_all = np.concatenate(x_all); w_all = np.concatenate(w_all)
    M_all = np.concatenate(M_all); kappa_all = np.concatenate(kappa_all)
    eps0_all = np.concatenate(eps0_all); theta_all = np.concatenate(theta_all)
    V_all = np.concatenate(V_all); q_all = np.concatenate(q_all)
    N_all = np.concatenate(N_all); M_sec_all = np.concatenate(M_sec_all)

    # Support moments
    print(f"\n  Analytical support moment: {M_sup/1e6:.2f} kN·m")
    for j in range(n_spans - 1):
        with torch.no_grad():
            Mr = span_nets[j](xi_bcs[j])["M_bar"][1].item() * scales.M_ref
            Ml = span_nets[j+1](xi_bcs[j+1])["M_bar"][0].item() * scales.M_ref
        print(f"  Support {j+1}: left={Mr/1e6:.2f}, right={Ml/1e6:.2f} kN·m")

    # MSE
    w_ri = np.interp(x_all, x_ref, w_ref)
    M_ri = np.interp(x_all, x_ref, M_ref)
    nrmse_w = np.sqrt(np.mean((w_all - w_ri)**2)) / np.ptp(w_ri) * 100
    nrmse_M = np.sqrt(np.mean((M_all - M_ri)**2)) / np.ptp(M_ri) * 100
    print(f"\n  NRMSE_w = {nrmse_w:.2f}%")
    print(f"  NRMSE_M = {nrmse_M:.2f}%")
    print(f"  max|N| = {np.max(np.abs(N_all)):.0f} N")

    # ── Compute top/bottom strains ──
    h = cfg.section_height
    y_top, y_bot = h/2, -h/2
    eps_top_p = eps0_all - kappa_all * y_top
    eps_bot_p = eps0_all - kappa_all * y_bot
    kappa_ri = np.interp(x_all, x_ref, kappa_ref)
    eps0_ri = np.interp(x_all, x_ref, eps0_ref)
    eps_top_r = eps0_ri - kappa_ri * y_top
    eps_bot_r = eps0_ri - kappa_ri * y_bot

    # ── Plot 5x4 ──
    fig, axes = plt.subplots(5, 4, figsize=(24, 26))
    def add_sup(ax):
        for sx in sup_x: ax.axvline(sx, color="gray", ls=":", lw=0.5)

    # Row 0: w, theta, kappa, eps0
    ax = axes[0,0]; ax.plot(x_all, w_all, "b-", lw=2, label="PINN"); ax.plot(x_ref, w_ref, "r--", lw=1.5, label="Ref")
    add_sup(ax); ax.set_title("Displacement w"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0,1]; ax.plot(x_all, theta_all, "b-", lw=2, label="PINN"); ax.plot(x_ref, theta_ref, "r--", lw=1.5, label="Ref")
    add_sup(ax); ax.set_title("Rotation θ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0,2]; ax.plot(x_all, kappa_all, "b-", lw=2, label="PINN"); ax.plot(x_ref, kappa_ref, "r--", lw=1.5, label="Ref")
    add_sup(ax); ax.set_title("Curvature κ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0,3]; ax.plot(x_all, eps0_all, "b-", lw=2, label="PINN"); ax.plot(x_ref, eps0_ref, "r--", lw=1.5, label="Ref")
    add_sup(ax); ax.set_title("Centroidal strain ε₀"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 1: M, V, q, N
    ax = axes[1,0]; ax.plot(x_all, M_all/1e6, "b-", lw=2, label="M_net"); ax.plot(x_all, M_sec_all/1e6, "g-.", lw=1, label="M_sec")
    ax.plot(x_ref, M_ref/1e6, "r--", lw=1.5, label="Ref"); add_sup(ax)
    ax.set_title("Bending moment M"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1,1]; ax.plot(x_all, V_all/1e3, "b-", lw=2, label="PINN"); ax.plot(x_ref, V_ref/1e3, "r--", lw=1.5, label="Ref")
    add_sup(ax); ax.set_title("Shear V"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1,2]; ax.plot(x_all, q_all, "b-", lw=2, label="PINN -M''"); ax.axhline(q, color="r", ls="--", lw=1.5, label=f"q={q}")
    add_sup(ax); ax.set_title("Equilibrium -M''"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1,3]; ax.plot(x_all, N_all/1e3, "b-", lw=2, label="N_sec"); ax.axhline(0, color="r", ls="--", lw=1)
    add_sup(ax); ax.set_title("Axial force N"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 2: M-κ, ε₀-κ, ε_top-κ, ε_bot-κ
    ax = axes[2,0]; ax.plot(kappa_all, M_all/1e6, "b.", ms=2, label="PINN"); ax.plot(kappa_ri, M_ri/1e6, "r+", ms=3, label="Ref")
    ax.set_xlabel("κ"); ax.set_ylabel("M (kN·m)"); ax.set_title("M-κ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2,1]; ax.plot(kappa_all, eps0_all*1e3, "b.", ms=2, label="PINN"); ax.plot(kappa_ri, eps0_ri*1e3, "r+", ms=3, label="Ref")
    ax.set_xlabel("κ"); ax.set_ylabel("ε₀ (‰)"); ax.set_title("ε₀-κ"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2,2]; ax.plot(kappa_all, eps_top_p*1e3, "b.", ms=2, label="PINN"); ax.plot(kappa_ri, eps_top_r*1e3, "r+", ms=3, label="Ref")
    ax.set_xlabel("κ"); ax.set_ylabel("ε_top (‰)"); ax.set_title(f"ε_top-κ (y={y_top:.0f})"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2,3]; ax.plot(kappa_all, eps_bot_p*1e3, "b.", ms=2, label="PINN"); ax.plot(kappa_ri, eps_bot_r*1e3, "r+", ms=3, label="Ref")
    ax.set_xlabel("κ"); ax.set_ylabel("ε_bot (‰)"); ax.set_title(f"ε_bot-κ (y={y_bot:.0f})"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 3: all loss components, lr, error
    ax = axes[3,0]; ax.semilogy(loss_history, "k-", lw=1, label="Total")
    for name, hist in comp_history.items():
        ax.semilogy(hist, lw=0.7, alpha=0.7, label=name)
    ax.set_title("All losses"); ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[3,1]; ax.semilogy(comp_history.get("compat_M", [1]), lw=1, label="compat_M")
    ax.semilogy(comp_history.get("compat_θ", [1]), lw=1, label="compat_θ")
    ax.set_title("Compat losses"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[3,2]
    ax2 = ax.twinx()
    ax.plot(lr_history, "k-", lw=1, label="lr")
    ax.set_ylabel("Learning rate"); ax.set_title("LR & weights")
    for name, hist in weight_history.items():
        ax2.semilogy(hist, lw=0.7, alpha=0.7, label=name)
    ax2.set_ylabel("Effective weight")
    ax2.legend(fontsize=4, ncol=2, loc="center right")

    ax = axes[3,3]
    ax.bar(["NRMSE_w", "NRMSE_M"], [nrmse_w, nrmse_M])
    for j, v in enumerate([nrmse_w, nrmse_M]): ax.text(j, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title("Error"); ax.grid(True, alpha=0.3, axis="y")

    # Row 4: stress contour
    n_y = 50; y_arr = np.linspace(-h/2, h/2, n_y)
    X_grid, Y_grid = np.meshgrid(x_all, y_arr)
    stress_pinn = np.zeros_like(X_grid)
    stress_ref_grid = np.zeros_like(X_grid)
    for j, yj in enumerate(y_arr):
        strain_p = eps0_all - kappa_all * yj
        strain_r = eps0_ri - kappa_ri * yj
        with torch.no_grad():
            stress_pinn[j] = concrete.stress(torch.tensor(strain_p, dtype=torch.float32)).numpy()
            stress_ref_grid[j] = concrete.stress(torch.tensor(strain_r, dtype=torch.float32)).numpy()

    vmin = min(stress_pinn.min(), stress_ref_grid.min(), -30)
    vmax = max(stress_pinn.max(), stress_ref_grid.max(), 5)
    for ax_s, sgrid, title in [(axes[4,0], stress_pinn, "PINN σ"), (axes[4,1], stress_ref_grid, "Ref σ")]:
        im = ax_s.pcolormesh(X_grid, Y_grid, sgrid, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        ax_s.contour(X_grid, Y_grid, sgrid, levels=[0], colors="k", linewidths=1.5)
        for sx in sup_x: ax_s.axvline(sx, color="gray", ls=":", lw=0.5)
        ax_s.set_title(title); ax_s.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax_s, shrink=0.7)
    axes[4,1].set_xlabel("x (mm)")
    axes[4,2].axis("off"); axes[4,3].axis("off")

    fig.suptitle(f"3-span: L={L_span}×{n_spans}, q={q} | NRMSE_w={nrmse_w:.1f}%, M={nrmse_M:.1f}%", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "continuous_beam.png"), dpi=150)
    plt.close(fig)
    # ── Save run summary ──
    sup_moments = []
    for j in range(n_spans - 1):
        with torch.no_grad():
            Mr = span_nets[j](xi_bcs[j])["M_bar"][1].item() * scales.M_ref
            Ml = span_nets[j+1](xi_bcs[j+1])["M_bar"][0].item() * scales.M_ref
        sup_moments.append((Mr, Ml))

    summary = f"""Run: {os.path.basename(RUN_DIR)}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

=== Configuration ===
Type: {n_spans}-span continuous beam
Span length: {L_span}mm, Total: {L_total}mm
Load: q={q}N/mm, N_applied={cfg.N_applied}N
Section: {cfg.section_width}x{cfg.section_height}mm
Material: elastic (Ec={cfg.Ec}, Es=200000)
Rebar: {cfg.rebar_layout}

=== PINN Architecture ===
Networks: {n_spans} × 3-MLP (w, eps0, M), N from fiber section
Hidden: {cfg.hidden_dims}, activation={cfg.activation}
Norm coeffs: w={nc['w']:.3e}, M={nc['M']:.3e}, eps0={nc['eps0']:.3e}

=== Training ===
Epochs: {cfg.n_epochs}, lr={cfg.learning_rate}, scheduler=None (constant)
Collocation: {cfg.n_collocation} per span
Loss weights: {dict(cfg.loss_weights)}
M_net_bc: disabled (M=0 only at beam ends, w_Mbc={w_Mbc})
Compat weight: {w_compat} (M + θ continuity)
Warmup: M_end_bc + compat ramp over first {cfg.n_epochs//2} epochs

=== Results ===
NRMSE_w = {nrmse_w:.2f}%
NRMSE_M = {nrmse_M:.2f}%
max|N| = {np.max(np.abs(N_all)):.0f} N

Support moments (analytical: {M_sup/1e6:.2f} kN·m):
"""
    for j, (Mr, Ml) in enumerate(sup_moments):
        summary += f"  Support {j+1}: left={Mr/1e6:.2f}, right={Ml/1e6:.2f} kN·m\n"

    with open(os.path.join(RUN_DIR, "run_summary.txt"), "w") as f:
        f.write(summary)

    print(f"\n  Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
