#!/usr/bin/env python3
"""Test and debug bisection-based nonlinear beam solver.

Step 1: Test section response at known strain states
Step 2: Test eps0 bisection for N=0 at various kappa
Step 3: Test kappa bisection for M=M_target
Step 4: Full beam solve at multiple load levels, plot results

Usage:
    cd pinn
    python -m examples.test_bisection
"""

import os
import sys
import numpy as np
import torch
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection

OUT_DIR = "outputs/bisection_debug"
os.makedirs(OUT_DIR, exist_ok=True)


def build_section():
    concrete = ManderConcrete(fc=30.0, Ec=25000.0, eps_co=-0.002, eps_cu=-0.004,
                              Gf=0.1, h=150.0)
    steel = BilinearSteel(fy=400.0, Es=200000.0, b=0.01)
    rc = RCRectSection(
        width=300.0, height=500.0,
        concrete=concrete, steel=steel,
        n_concrete_fibers=20,
        rebar_layout=[(-200.0, 1520.0), (200.0, 804.0)],
    )
    return rc


def section_NM(section, eps0, kappa):
    """Scalar eps0, kappa → (N, M)."""
    resp = section.response(torch.tensor([eps0]), torch.tensor([kappa]))
    return resp["N"].item(), resp["M"].item()


# ==================================================================
# Step 1: Moment-curvature curve at N=0 (sweep kappa, bisect eps0)
# ==================================================================

def find_eps0_for_N_zero(section, kappa, lo=-0.01, hi=0.01, tol=1.0, max_iter=80):
    N_lo, _ = section_NM(section, lo, kappa)
    N_hi, _ = section_NM(section, hi, kappa)
    for _ in range(10):
        if N_lo * N_hi <= 0:
            break
        lo *= 2; hi *= 2
        N_lo, _ = section_NM(section, lo, kappa)
        N_hi, _ = section_NM(section, hi, kappa)
    if N_lo * N_hi > 0:
        return 0.0, False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        N_mid, _ = section_NM(section, mid, kappa)
        if abs(N_mid) < tol:
            return mid, True
        if N_mid * N_lo < 0:
            hi = mid
        else:
            lo = mid; N_lo = N_mid
    return (lo + hi) / 2.0, True


def step1_moment_curvature(section):
    """Sweep kappa, find eps0 for N=0, record M-kappa curve."""
    print("=" * 60)
    print("  Step 1: Moment-Curvature at N=0")
    print("=" * 60)

    kappas = np.linspace(0, 8e-5, 200)
    M_list, eps0_list, N_list = [], [], []

    for kap in kappas:
        eps0, ok = find_eps0_for_N_zero(section, kap)
        N, M = section_NM(section, eps0, kap)
        M_list.append(M)
        eps0_list.append(eps0)
        N_list.append(N)

    M_arr = np.array(M_list)
    eps0_arr = np.array(eps0_list)
    N_arr = np.array(N_list)

    print(f"  kappa range: [0, {kappas[-1]:.2e}]")
    print(f"  M range:     [{M_arr.min():.0f}, {M_arr.max():.0f}] N.mm")
    print(f"  eps0 range:  [{eps0_arr.min():.6f}, {eps0_arr.max():.6f}]")
    print(f"  max|N|:      {np.max(np.abs(N_arr)):.1f} N")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(kappas * 1e6, M_arr / 1e6, "b-", linewidth=2)
    ax.set_xlabel("Curvature (1e-6 /mm)")
    ax.set_ylabel("Moment (kN.m)")
    ax.set_title("M - kappa (N=0)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(kappas * 1e6, eps0_arr * 1e3, "r-", linewidth=2)
    ax.set_xlabel("Curvature (1e-6 /mm)")
    ax.set_ylabel("eps0 (x1e-3)")
    ax.set_title("eps0 at N=0")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(kappas * 1e6, N_arr, "g-", linewidth=2)
    ax.set_xlabel("Curvature (1e-6 /mm)")
    ax.set_ylabel("N (N)")
    ax.set_title("N residual")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step1_M_kappa.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/step1_M_kappa.png")

    return kappas, M_arr, eps0_arr


# ==================================================================
# Step 2: Bisect kappa for target M
# ==================================================================

def find_kappa_for_M(section, M_target, kappa_data, M_data, tol=100.0, max_iter=80):
    """Bisect kappa so that M(eps0_balanced, kappa) = M_target.

    Uses the M-kappa data to set initial bounds.
    """
    if abs(M_target) < tol:
        return 0.0, 0.0

    sign = 1.0 if M_target > 0 else -1.0

    # Use M-kappa data to find bracket
    kappa_lo = 0.0
    kappa_hi = kappa_data[-1]  # max kappa from sweep

    for _ in range(max_iter):
        kappa_mid = (kappa_lo + kappa_hi) / 2.0
        eps0_mid, _ = find_eps0_for_N_zero(section, kappa_mid * sign)
        _, M_mid = section_NM(section, eps0_mid, kappa_mid * sign)

        if abs(M_mid - M_target) < tol:
            return eps0_mid, kappa_mid * sign

        if abs(M_mid) < abs(M_target):
            kappa_lo = kappa_mid
        else:
            kappa_hi = kappa_mid

    kappa_final = (kappa_lo + kappa_hi) / 2.0 * sign
    eps0_final, _ = find_eps0_for_N_zero(section, kappa_final)
    return eps0_final, kappa_final


def step2_test_bisect_M(section, kappa_data, M_data):
    """Test M bisection at several target values."""
    print("\n" + "=" * 60)
    print("  Step 2: Test kappa bisection for target M")
    print("=" * 60)

    M_max = M_data.max()
    targets = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    print(f"  {'M_target':>12s}  {'eps0':>10s}  {'kappa':>12s}  {'N':>10s}  {'M_got':>12s}  {'err%':>8s}")
    for frac in targets:
        Mt = M_max * frac
        eps0, kap = find_kappa_for_M(section, Mt, kappa_data, M_data)
        N, M = section_NM(section, eps0, kap)
        err = abs(M - Mt) / Mt * 100
        print(f"  {Mt:12.0f}  {eps0:10.6f}  {kap:12.6e}  {N:10.1f}  {M:12.0f}  {err:7.3f}%")


# ==================================================================
# Step 3: Full beam solve at multiple load levels
# ==================================================================

def solve_beam(section, L, q, n_pts, kappa_data, M_data):
    """Solve simply supported beam: M(x)=q/2*x*(L-x), bisect for kappa, integrate for w."""
    from scipy.integrate import cumulative_trapezoid

    x = np.linspace(0, L, n_pts)
    M_target = q / 2.0 * x * (L - x)

    eps0_arr = np.zeros(n_pts)
    kappa_arr = np.zeros(n_pts)
    N_arr = np.zeros(n_pts)

    for i in range(n_pts):
        if M_target[i] < 100.0:
            continue
        eps0_i, kap_i = find_kappa_for_M(section, M_target[i], kappa_data, M_data)
        eps0_arr[i] = eps0_i
        kappa_arr[i] = kap_i
        N_i, _ = section_NM(section, eps0_i, kap_i)
        N_arr[i] = N_i

    # Integrate: w'' = -kappa → w
    theta = cumulative_trapezoid(-kappa_arr, x, initial=0.0)
    w_raw = cumulative_trapezoid(theta, x, initial=0.0)
    # BC: w(0)=0, w(L)=0
    wp0 = -w_raw[-1] / L
    w = wp0 * x + w_raw

    return {"x": x, "w": w, "M": M_target, "N": N_arr, "eps0": eps0_arr, "kappa": kappa_arr}


def step3_multi_load(section, kappa_data, M_data):
    """Solve beam at multiple load levels, plot comparison."""
    print("\n" + "=" * 60)
    print("  Step 3: Beam solutions at multiple load levels")
    print("=" * 60)

    L = 3000.0
    n_pts = 80
    loads = [5.0, 10.0, 20.0, 40.0]  # N/mm

    results = {}
    for q in loads:
        print(f"  q = {q:.0f} N/mm ...")
        res = solve_beam(section, L, q, n_pts, kappa_data, M_data)
        results[q] = res
        mid = n_pts // 2
        print(f"    midspan: w={res['w'][mid]:.4f} mm, M={res['M'][mid]:.0f} N.mm, "
              f"kappa={res['kappa'][mid]:.6e}, eps0={res['eps0'][mid]:.6f}, N={res['N'][mid]:.0f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for q, res in results.items():
        x = res["x"]
        label = f"q={q:.0f}"

        axes[0, 0].plot(x, res["w"], linewidth=1.5, label=label)
        axes[0, 1].plot(x, res["eps0"] * 1e3, linewidth=1.5, label=label)
        axes[0, 2].plot(x, res["kappa"] * 1e6, linewidth=1.5, label=label)
        axes[1, 0].plot(x, res["M"] / 1e6, linewidth=1.5, label=label)
        axes[1, 1].plot(x, res["N"], linewidth=1.5, label=label)

    titles = ["Displacement w (mm)", "Centroidal strain eps0 (x1e-3)",
              "Curvature (1e-6 /mm)", "Moment M (kN.m)", "Axial force N (N)"]
    ylabels = ["w (mm)", "eps0 (x1e-3)", "kappa (1e-6 /mm)", "M (kN.m)", "N (N)"]

    for i, ax in enumerate(axes.flat[:5]):
        ax.set_xlabel("x (mm)")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Last subplot: M-kappa comparison from section sweep
    ax = axes[1, 2]
    kappas_plot = kappa_data * 1e6
    ax.plot(kappas_plot, M_data / 1e6, "k-", linewidth=2, label="Section M-kappa")
    for q, res in results.items():
        mid = len(res["x"]) // 2
        ax.plot(res["kappa"][mid] * 1e6, res["M"][mid] / 1e6, "o", markersize=8,
                label=f"q={q:.0f} midspan")
    ax.set_xlabel("Curvature (1e-6 /mm)")
    ax.set_ylabel("M (kN.m)")
    ax.set_title("M-kappa curve + midspan points")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step3_multi_load.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {OUT_DIR}/step3_multi_load.png")


# ==================================================================
# Main
# ==================================================================

if __name__ == "__main__":
    rc = build_section()
    section = rc.section

    kappa_data, M_data, eps0_data = step1_moment_curvature(section)
    step2_test_bisect_M(section, kappa_data, M_data)
    step3_multi_load(section, kappa_data, M_data)

    print("\n" + "=" * 60)
    print(f"  ALL DONE — results in {os.path.abspath(OUT_DIR)}")
    print("=" * 60)
