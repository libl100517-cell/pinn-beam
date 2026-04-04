#!/usr/bin/env python3
"""Simply supported RC beam — forward and inverse PINN analysis.

This script demonstrates:
1. Elastic forward analysis
2. Elasto-plastic forward analysis
3. Inverse parameter identification from synthetic displacement data

Each run creates a new output directory: outputs/run_001, run_002, ...

Usage:
    cd pinn
    python -m examples.run_simply_supported_beam
"""

import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.simply_supported_elastic import get_config as elastic_config
from configs.simply_supported_elastoplastic import get_config as elastoplastic_config
from configs.simply_supported_inverse import get_config as inverse_config
from materials import ManderConcrete, BilinearSteel
from solvers import ForwardSolver, InverseSolver
from plotting import PlotResults
from utils import set_seed


def make_run_dir(base: str = "outputs") -> str:
    """Create outputs/run_NNN with auto-incrementing number."""
    os.makedirs(base, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(base, "run_[0-9][0-9][0-9]")))
    if existing:
        last_num = int(os.path.basename(existing[-1]).split("_")[1])
        next_num = last_num + 1
    else:
        next_num = 1
    run_dir = os.path.join(base, f"run_{next_num:03d}")
    os.makedirs(run_dir)
    return run_dir


# Create output directory for this run
RUN_DIR = make_run_dir()
plotter = PlotResults(save_dir=RUN_DIR)
print(f">>> Output directory: {os.path.abspath(RUN_DIR)}")


def save_metadata(run_dir: str, extra: dict | None = None):
    """Save run metadata to JSON."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    if extra:
        meta.update(extra)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def run_elastic_forward():
    """Run elastic forward analysis on a simply supported beam."""
    print("=" * 60)
    print("  ELASTIC FORWARD ANALYSIS")
    print("=" * 60)

    cfg = elastic_config()
    solver = ForwardSolver(cfg)
    results = solver.solve()

    # Predict fields
    xi_plot = torch.linspace(0, 1, 100).unsqueeze(1)
    fields = results["model"].predict(xi_plot)
    fields_np = {k: v.numpy() for k, v in fields.items()}

    # Plot
    fig = plotter.plot_fields(xi_plot.numpy(), fields_np, results["scales"], "Elastic — ")
    fig.savefig(os.path.join(RUN_DIR, "elastic_fields.png"), dpi=150)
    plt.close(fig)

    fig = plotter.plot_loss_history(results["logger"])
    fig.savefig(os.path.join(RUN_DIR, "elastic_loss.png"), dpi=150)
    plt.close(fig)

    # Material curves
    concrete = ManderConcrete(fc=cfg.fc, Ec=cfg.Ec)
    steel = BilinearSteel(fy=cfg.fy, Es=cfg.Es)
    fig = plotter.plot_concrete_stress_strain(concrete)
    fig.savefig(os.path.join(RUN_DIR, "concrete_mander.png"), dpi=150)
    plt.close(fig)
    fig = plotter.plot_steel_stress_strain(steel)
    fig.savefig(os.path.join(RUN_DIR, "steel_bilinear.png"), dpi=150)
    plt.close(fig)

    print(f"\nElastic forward analysis complete. Figures saved to {RUN_DIR}/")
    return results


def run_elastoplastic_forward():
    """Run elasto-plastic forward analysis."""
    print("\n" + "=" * 60)
    print("  ELASTO-PLASTIC FORWARD ANALYSIS")
    print("=" * 60)

    cfg = elastoplastic_config()
    solver = ForwardSolver(cfg)
    results = solver.solve()

    xi_plot = torch.linspace(0, 1, 100).unsqueeze(1)
    fields = results["model"].predict(xi_plot)
    fields_np = {k: v.numpy() for k, v in fields.items()}

    fig = plotter.plot_fields(xi_plot.numpy(), fields_np, results["scales"], "Elasto-plastic — ")
    fig.savefig(os.path.join(RUN_DIR, "elastoplastic_fields.png"), dpi=150)
    plt.close(fig)

    fig = plotter.plot_loss_history(results["logger"])
    fig.savefig(os.path.join(RUN_DIR, "elastoplastic_loss.png"), dpi=150)
    plt.close(fig)

    print(f"\nElasto-plastic forward analysis complete.")
    return results


def run_inverse():
    """Run inverse parameter identification using synthetic data."""
    print("\n" + "=" * 60)
    print("  INVERSE ANALYSIS — Ec IDENTIFICATION")
    print("=" * 60)

    # Step 1: Generate synthetic data using the forward solver with true params
    true_Ec = 25000.0
    true_cfg = elastic_config()
    true_cfg.Ec = true_Ec
    true_solver = ForwardSolver(true_cfg)
    true_results = true_solver.solve()

    # Sample observations at 20 interior points
    n_obs = 20
    xi_obs = torch.linspace(0.05, 0.95, n_obs).unsqueeze(1)
    true_fields = true_results["model"].predict(xi_obs)
    w_obs = true_fields["w_bar"]

    # Add noise (optional)
    set_seed(123)
    noise_level = 0.01  # 1% noise
    noise = noise_level * w_obs.abs().max() * torch.randn_like(w_obs)
    w_obs_noisy = w_obs + noise

    # Step 2: Run inverse analysis with wrong initial guess for Ec
    inv_cfg = inverse_config()
    trainable = {
        "Ec": {"init": 18000.0, "bounds": (10000.0, 50000.0)},
    }

    inv_solver = InverseSolver(
        config=inv_cfg,
        trainable_params=trainable,
        observation_xi=xi_obs,
        observation_w=w_obs_noisy,
    )
    inv_results = inv_solver.solve()

    # Report results
    identified = inv_results["identified_params"]
    print(f"\n--- Inverse Results ---")
    print(f"  True Ec       = {true_Ec:.1f} MPa")
    print(f"  Identified Ec = {identified['Ec']:.1f} MPa")
    print(f"  Error         = {abs(identified['Ec'] - true_Ec) / true_Ec * 100:.2f}%")

    # Save inverse results to JSON
    inv_summary = {
        "true_Ec": true_Ec,
        "identified_Ec": identified["Ec"],
        "error_pct": abs(identified["Ec"] - true_Ec) / true_Ec * 100,
        "noise_level": noise_level,
    }
    with open(os.path.join(RUN_DIR, "inverse_results.json"), "w") as f:
        json.dump(inv_summary, f, indent=2)

    # Plots
    fig = plotter.plot_loss_history(inv_results["logger"])
    fig.savefig(os.path.join(RUN_DIR, "inverse_loss.png"), dpi=150)
    plt.close(fig)

    fig = plotter.plot_param_convergence(
        inv_results["logger"],
        true_values={"Ec": true_Ec},
    )
    fig.savefig(os.path.join(RUN_DIR, "inverse_Ec_convergence.png"), dpi=150)
    plt.close(fig)

    # Prediction vs observation
    xi_dense = torch.linspace(0, 1, 100).unsqueeze(1)
    pred_fields = inv_results["model"].predict(xi_dense)
    scales = inv_results["scales"]
    fig = plotter.plot_pred_vs_obs(
        x_obs=(xi_obs * scales.L_ref).numpy(),
        w_obs=(w_obs_noisy * scales.w_ref).numpy(),
        x_pred=(xi_dense * scales.L_ref).numpy(),
        w_pred=(pred_fields["w_bar"].numpy() * scales.w_ref),
    )
    fig.savefig(os.path.join(RUN_DIR, "inverse_pred_vs_obs.png"), dpi=150)
    plt.close(fig)

    print(f"\nInverse analysis complete. Figures saved to {RUN_DIR}/")
    return inv_results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    save_metadata(RUN_DIR)
    run_elastic_forward()
    run_elastoplastic_forward()
    run_inverse()
    print("\n" + "=" * 60)
    print(f"  ALL ANALYSES COMPLETE — results in {os.path.abspath(RUN_DIR)}")
    print("=" * 60)
