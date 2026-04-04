#!/usr/bin/env python3
"""Simply supported RC beam — forward PINN analysis.

Plots PINN predictions alongside analytical/numerical reference solutions.

Usage:
    cd pinn
    python -m examples.run_forward
"""

import os
import sys
import glob
import json
from datetime import datetime

# Allow running as script: python examples/run_forward.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.simply_supported_elastic import get_config as elastic_config
from configs.simply_supported_elastoplastic import get_config as elastoplastic_config
from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection
from solvers import ForwardSolver
from solvers.section_analysis import solve_beam_nonlinear
from plotting import PlotResults
from utils import set_seed


def make_run_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(base, "run_[0-9][0-9][0-9]")))
    next_num = int(os.path.basename(existing[-1]).split("_")[1]) + 1 if existing else 1
    run_dir = os.path.join(base, f"run_{next_num:03d}")
    os.makedirs(run_dir)
    return run_dir


RUN_DIR = make_run_dir()
plotter = PlotResults(save_dir=RUN_DIR)
print(f">>> Output directory: {os.path.abspath(RUN_DIR)}")


def save_metadata(run_dir: str):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_section(cfg, elastic: bool):
    if elastic:
        concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=-0.1, eps_cu=-0.2,
                                  Gf=cfg.Gf, h=cfg.concrete_h)
        steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=1.0)
    else:
        concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, eps_co=cfg.eps_co, eps_cu=cfg.eps_cu,
                                  Gf=cfg.Gf, h=cfg.concrete_h)
        steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es, b=cfg.steel_b)
    return RCRectSection(
        width=cfg.section_width, height=cfg.section_height,
        concrete=concrete, steel=steel,
        n_concrete_fibers=cfg.n_concrete_fibers, rebar_layout=cfg.rebar_layout,
    )


def compute_elastic_exact(cfg, x: np.ndarray):
    """Analytical solution for elastic simply supported beam."""
    L, q = cfg.beam_length, cfg.q
    rc = build_section(cfg, elastic=True)
    EI = sum(
        (f.material.Es if isinstance(f.material, BilinearSteel) else f.material.Ec)
        * f.area * f.y ** 2
        for f in rc.section.fibers.fibers
    )
    M = q / 2.0 * x * (L - x)
    w = q / (24.0 * EI) * x * (L**3 - 2*L*x**2 + x**3)
    return {"x": x, "w": w, "M": M, "N": np.zeros_like(x),
            "eps0": np.zeros_like(x), "kappa": M / EI, "EI": EI}


def compute_nonlinear_ref(cfg, n_pts: int = 100):
    """Numerical reference via bisection."""
    rc = build_section(cfg, elastic=False)
    return solve_beam_nonlinear(rc.section, cfg.beam_length, cfg.q, n_pts)


def plot_fields_vs_ref(xi_np, fields_np, scales, ref, title_prefix, save_path):
    """Plot PINN (blue solid) vs Reference (red dashed), 4 subplots."""
    x_pinn = xi_np.flatten() * scales.L_ref
    x_ref = ref["x"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Displacement — PINN w is negative-downward, ref w is positive-downward
    ax = axes[0, 0]
    w_pinn = fields_np["w_bar"].flatten() * scales.w_ref
    ax.plot(x_pinn, w_pinn, "b-", linewidth=2, label="PINN")
    ax.plot(x_ref, -ref["w"], "r--", linewidth=1.5, label="Reference")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("w (mm)")
    ax.set_title(f"{title_prefix}Displacement"); ax.legend(); ax.grid(True, alpha=0.3)

    # Centroidal strain
    ax = axes[0, 1]
    ax.plot(x_pinn, fields_np["eps0_bar"].flatten() * scales.eps_ref, "b-", linewidth=2, label="PINN")
    ax.plot(x_ref, ref["eps0"], "r--", linewidth=1.5, label="Reference")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("eps0")
    ax.set_title(f"{title_prefix}Centroidal strain"); ax.legend(); ax.grid(True, alpha=0.3)

    # Bending moment
    ax = axes[1, 0]
    ax.plot(x_pinn, fields_np["M_bar"].flatten() * scales.M_ref, "b-", linewidth=2, label="PINN")
    ax.plot(x_ref, ref["M"], "r--", linewidth=1.5, label="Reference")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("M (N.mm)")
    ax.set_title(f"{title_prefix}Bending moment"); ax.legend(); ax.grid(True, alpha=0.3)

    # Axial force
    ax = axes[1, 1]
    ax.plot(x_pinn, fields_np["N_bar"].flatten() * scales.F_ref, "b-", linewidth=2, label="PINN")
    ax.plot(x_ref, ref["N"], "r--", linewidth=1.5, label="Reference")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("N (N)")
    ax.set_title(f"{title_prefix}Axial force"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_error(label, fields_np, scales, ref):
    w_pinn = fields_np["w_bar"].flatten() * scales.w_ref
    w_ref = -ref["w"]
    M_pinn = fields_np["M_bar"].flatten() * scales.M_ref
    mid = len(w_pinn) // 2
    mid_r = len(ref["x"]) // 2
    w_err = abs(w_pinn[mid] - w_ref[mid_r]) / max(abs(w_ref[mid_r]), 1e-12) * 100
    M_err = abs(M_pinn[mid] - ref["M"][mid_r]) / max(abs(ref["M"][mid_r]), 1e-12) * 100
    print(f"\n  {label} midspan:")
    print(f"    w:  PINN={w_pinn[mid]:.4f} mm, Ref={w_ref[mid_r]:.4f} mm, Err={w_err:.2f}%")
    print(f"    M:  PINN={M_pinn[mid]:.0f}, Ref={ref['M'][mid_r]:.0f} N.mm, Err={M_err:.2f}%")


def run_elastic():
    print("=" * 60)
    print("  ELASTIC FORWARD ANALYSIS")
    print("=" * 60)

    cfg = elastic_config()
    solver = ForwardSolver(cfg, log_dir=RUN_DIR)
    results = solver.solve()

    xi_plot = torch.linspace(0, 1, 100).unsqueeze(1)
    fields = results["model"].predict(xi_plot)
    fields_np = {k: v.numpy() for k, v in fields.items()}
    scales = results["scales"]

    ref = compute_elastic_exact(cfg, xi_plot.numpy().flatten() * scales.L_ref)
    plot_fields_vs_ref(xi_plot.numpy(), fields_np, scales, ref,
                       "Elastic — ", os.path.join(RUN_DIR, "elastic_fields.png"))

    fig = plotter.plot_loss_history(results["logger"])
    fig.savefig(os.path.join(RUN_DIR, "elastic_loss.png"), dpi=150)
    plt.close(fig)

    # Material curves
    concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec, Gf=cfg.Gf, h=cfg.concrete_h)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es)
    fig = plotter.plot_concrete_stress_strain(concrete)
    fig.savefig(os.path.join(RUN_DIR, "concrete_mander.png"), dpi=150)
    plt.close(fig)
    fig = plotter.plot_steel_stress_strain(steel)
    fig.savefig(os.path.join(RUN_DIR, "steel_bilinear.png"), dpi=150)
    plt.close(fig)

    print_error("Elastic", fields_np, scales, ref)
    print(f"  EI = {ref['EI']:.3e} N.mm^2")
    print(f"\nElastic analysis complete.")
    return results


def run_elastoplastic():
    print("\n" + "=" * 60)
    print("  ELASTO-PLASTIC FORWARD ANALYSIS")
    print("=" * 60)

    cfg = elastoplastic_config()
    solver = ForwardSolver(cfg, log_dir=RUN_DIR)
    results = solver.solve()

    xi_plot = torch.linspace(0, 1, 100).unsqueeze(1)
    fields = results["model"].predict(xi_plot)
    fields_np = {k: v.numpy() for k, v in fields.items()}
    scales = results["scales"]

    ref = compute_nonlinear_ref(cfg, n_pts=100)
    plot_fields_vs_ref(xi_plot.numpy(), fields_np, scales, ref,
                       "Elasto-plastic — ", os.path.join(RUN_DIR, "elastoplastic_fields.png"))

    fig = plotter.plot_loss_history(results["logger"])
    fig.savefig(os.path.join(RUN_DIR, "elastoplastic_loss.png"), dpi=150)
    plt.close(fig)

    print_error("Elasto-plastic", fields_np, scales, ref)
    print(f"\nElasto-plastic analysis complete.")
    return results


if __name__ == "__main__":
    save_metadata(RUN_DIR)
    run_elastic()
    run_elastoplastic()
    print("\n" + "=" * 60)
    print(f"  DONE — results in {os.path.abspath(RUN_DIR)}")
    print("=" * 60)
