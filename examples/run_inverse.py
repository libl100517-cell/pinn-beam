#!/usr/bin/env python3
"""Simply supported RC beam — inverse PINN analysis.

Identifies Ec from synthetic displacement data.

Usage:
    cd pinn
    python -m examples.run_inverse
"""

import os
import sys
import glob
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.simply_supported_elastic import get_config as elastic_config
from configs.simply_supported_inverse import get_config as inverse_config
from solvers import ForwardSolver, InverseSolver
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


def run_inverse():
    print("=" * 60)
    print("  INVERSE ANALYSIS — Ec IDENTIFICATION")
    print("=" * 60)

    # Step 1: Generate synthetic data with true Ec
    true_Ec = 25000.0
    true_cfg = elastic_config()
    true_cfg.Ec = true_Ec
    print("Generating synthetic data with true parameters...")
    true_solver = ForwardSolver(true_cfg)
    true_results = true_solver.solve()

    # Sample observations
    n_obs = 20
    xi_obs = torch.linspace(0.05, 0.95, n_obs).unsqueeze(1)
    true_fields = true_results["model"].predict(xi_obs)
    w_obs = true_fields["w_bar"]

    # Add noise
    set_seed(123)
    noise_level = 0.01
    noise = noise_level * w_obs.abs().max() * torch.randn_like(w_obs)
    w_obs_noisy = w_obs + noise

    # Step 2: Inverse analysis
    inv_cfg = inverse_config()
    trainable = {
        "Ec": {"init": 18000.0, "bounds": (10000.0, 50000.0)},
    }

    print("Running inverse identification...")
    inv_solver = InverseSolver(
        config=inv_cfg,
        trainable_params=trainable,
        observation_xi=xi_obs,
        observation_w=w_obs_noisy,
    )
    inv_results = inv_solver.solve()

    # Report
    identified = inv_results["identified_params"]
    print(f"\n--- Inverse Results ---")
    print(f"  True Ec       = {true_Ec:.1f} MPa")
    print(f"  Identified Ec = {identified['Ec']:.1f} MPa")
    print(f"  Error         = {abs(identified['Ec'] - true_Ec) / true_Ec * 100:.2f}%")

    # Save results
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

    fig = plotter.plot_param_convergence(inv_results["logger"], true_values={"Ec": true_Ec})
    fig.savefig(os.path.join(RUN_DIR, "inverse_Ec_convergence.png"), dpi=150)
    plt.close(fig)

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

    print(f"\nInverse analysis complete. Results in {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    run_inverse()
