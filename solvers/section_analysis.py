"""Nonlinear section analysis via bisection.

For a simply supported beam under uniform load q, the moment is known:
    M(x) = q/2 * x * (L - x)

At each section, find (eps0, kappa) satisfying N=0, M=M_target.
Then integrate curvature twice to get displacement.
"""

import numpy as np
import torch
from typing import Dict, Tuple
from scipy.integrate import cumulative_trapezoid

from sections.fiber_section import FiberSection


def _section_NM(section: FiberSection, eps0: float, kappa: float) -> Tuple[float, float]:
    """Evaluate section N and M for scalar eps0, kappa."""
    resp = section.response(torch.tensor([eps0]), torch.tensor([kappa]))
    return resp["N"].item(), resp["M"].item()


def _find_eps0_for_N_zero(
    section: FiberSection,
    kappa: float,
    tol: float = 1.0,
    max_iter: int = 80,
) -> float:
    """Bisect eps0 so that N(eps0, kappa) ≈ 0."""
    lo, hi = -0.01, 0.01
    N_lo, _ = _section_NM(section, lo, kappa)
    N_hi, _ = _section_NM(section, hi, kappa)
    for _ in range(10):
        if N_lo * N_hi <= 0:
            break
        lo *= 2; hi *= 2
        N_lo, _ = _section_NM(section, lo, kappa)
        N_hi, _ = _section_NM(section, hi, kappa)
    if N_lo * N_hi > 0:
        return 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        N_mid, _ = _section_NM(section, mid, kappa)
        if abs(N_mid) < tol:
            return mid
        if N_mid * N_lo < 0:
            hi = mid
        else:
            lo = mid; N_lo = N_mid
    return (lo + hi) / 2.0


def build_M_kappa_curve(
    section: FiberSection,
    kappa_max: float = 8e-5,
    n_pts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep kappa ∈ [0, kappa_max], bisect eps0 for N=0, return (kappa, M) arrays."""
    kappas = np.linspace(0, kappa_max, n_pts)
    M_arr = np.zeros(n_pts)
    for i, kap in enumerate(kappas):
        eps0 = _find_eps0_for_N_zero(section, kap)
        _, M = _section_NM(section, eps0, kap)
        M_arr[i] = M
    return kappas, M_arr


def _find_kappa_for_M(
    section: FiberSection,
    M_target: float,
    kappa_max: float,
    tol: float = 100.0,
    max_iter: int = 80,
) -> Tuple[float, float]:
    """Bisect kappa ∈ [0, kappa_max] so that M(eps0_balanced, kappa) = M_target."""
    if abs(M_target) < tol:
        return 0.0, 0.0

    sign = 1.0 if M_target > 0 else -1.0
    kappa_lo, kappa_hi = 0.0, kappa_max

    for _ in range(max_iter):
        kappa_mid = (kappa_lo + kappa_hi) / 2.0
        eps0_mid = _find_eps0_for_N_zero(section, kappa_mid * sign)
        _, M_mid = _section_NM(section, eps0_mid, kappa_mid * sign)
        if abs(M_mid - M_target) < tol:
            return eps0_mid, kappa_mid * sign
        if abs(M_mid) < abs(M_target):
            kappa_lo = kappa_mid
        else:
            kappa_hi = kappa_mid

    kappa_final = (kappa_lo + kappa_hi) / 2.0 * sign
    eps0_final = _find_eps0_for_N_zero(section, kappa_final)
    return eps0_final, kappa_final


def solve_beam_nonlinear(
    section: FiberSection,
    L: float,
    q: float,
    n_pts: int = 100,
    kappa_max: float = 8e-5,
) -> Dict[str, np.ndarray]:
    """Solve simply supported beam with nonlinear section via bisection.

    Returns dict with dimensional numpy arrays: x, w, M, N, eps0, kappa.
    """
    x = np.linspace(0, L, n_pts)
    M_target = q / 2.0 * x * (L - x)

    eps0_arr = np.zeros(n_pts)
    kappa_arr = np.zeros(n_pts)
    N_arr = np.zeros(n_pts)

    print(f"  Solving nonlinear beam (bisection), q={q} N/mm ...")
    for i in range(n_pts):
        if M_target[i] < 100.0:
            continue
        eps0_i, kap_i = _find_kappa_for_M(section, M_target[i], kappa_max)
        eps0_arr[i] = eps0_i
        kappa_arr[i] = kap_i
        N_i, _ = _section_NM(section, eps0_i, kap_i)
        N_arr[i] = N_i

    # Integrate: w'' = -kappa → w
    theta = cumulative_trapezoid(-kappa_arr, x, initial=0.0)
    w_raw = cumulative_trapezoid(theta, x, initial=0.0)
    wp0 = -w_raw[-1] / L
    w = wp0 * x + w_raw

    mid = n_pts // 2
    print(f"  Done. midspan: w={w[mid]:.4f} mm, kappa={kappa_arr[mid]:.6e}, eps0={eps0_arr[mid]:.6f}")
    return {"x": x, "w": w, "M": M_target, "N": N_arr, "eps0": eps0_arr, "kappa": kappa_arr}
