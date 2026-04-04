"""Plotting utilities for PINN beam results."""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch

from physics.nondimensional import NonDimScales
from utils.logger import TrainingLogger


class PlotResults:
    """Collection of plotting routines for beam analysis results."""

    def __init__(self, save_dir: str = "figures"):
        self.save_dir = save_dir

    # ------------------------------------------------------------------
    # Field plots
    # ------------------------------------------------------------------

    def plot_fields(
        self,
        xi: np.ndarray,
        fields: Dict[str, np.ndarray],
        scales: NonDimScales,
        title_prefix: str = "",
    ) -> plt.Figure:
        """Plot displacement, strain, moment, and axial force distributions."""
        x = xi * scales.L_ref  # dimensional x

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Displacement
        ax = axes[0, 0]
        w_dim = fields["w_bar"] * scales.w_ref
        ax.plot(x, w_dim, "b-", linewidth=2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("w (mm)")
        ax.set_title(f"{title_prefix}Displacement")
        ax.grid(True, alpha=0.3)

        # Centroidal strain
        ax = axes[0, 1]
        ax.plot(x, fields["eps0_bar"] * scales.eps_ref, "r-", linewidth=2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("eps0")
        ax.set_title(f"{title_prefix}Centroidal strain")
        ax.grid(True, alpha=0.3)

        # Bending moment
        ax = axes[1, 0]
        M_dim = fields["M_bar"] * scales.M_ref
        ax.plot(x, M_dim, "g-", linewidth=2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("M (N.mm)")
        ax.set_title(f"{title_prefix}Bending moment")
        ax.grid(True, alpha=0.3)

        # Axial force
        ax = axes[1, 1]
        N_dim = fields["N_bar"] * scales.F_ref
        ax.plot(x, N_dim, "m-", linewidth=2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("N (N)")
        ax.set_title(f"{title_prefix}Axial force")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Material stress-strain curves
    # ------------------------------------------------------------------

    def plot_concrete_stress_strain(
        self,
        material,
        strain_range: tuple = (-0.004, 0.001),
        n_pts: int = 200,
    ) -> plt.Figure:
        """Plot Mander concrete stress-strain curve."""
        eps = torch.linspace(strain_range[0], strain_range[1], n_pts)
        sig = material.stress(eps).detach().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eps.numpy(), sig, "b-", linewidth=2)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("Mander Concrete Stress-Strain")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        fig.tight_layout()
        return fig

    def plot_steel_stress_strain(
        self,
        material,
        strain_range: tuple = (-0.01, 0.01),
        n_pts: int = 200,
    ) -> plt.Figure:
        """Plot bilinear steel stress-strain curve."""
        eps = torch.linspace(strain_range[0], strain_range[1], n_pts)
        sig = material.stress(eps).detach().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eps.numpy(), sig, "r-", linewidth=2)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("Bilinear Steel Stress-Strain")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_loss_history(self, logger: TrainingLogger) -> plt.Figure:
        """Plot training loss convergence."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(logger.loss_history, "k-", linewidth=1, label="Total")
        for name, hist in logger.component_history.items():
            if name != "total":
                ax.semilogy(hist, linewidth=0.8, alpha=0.7, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss History")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_param_convergence(self, logger: TrainingLogger, true_values: Dict[str, float] | None = None) -> plt.Figure:
        """Plot inverse parameter convergence."""
        n_params = len(logger.param_history)
        if n_params == 0:
            return plt.figure()

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]

        for ax, (name, hist) in zip(axes, logger.param_history.items()):
            ax.plot(hist, "b-", linewidth=1.5)
            if true_values and name in true_values:
                ax.axhline(true_values[name], color="r", linestyle="--", label=f"True = {true_values[name]:.1f}")
                ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(f"Parameter: {name}")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Prediction vs observation
    # ------------------------------------------------------------------

    def plot_pred_vs_obs(
        self,
        x_obs: np.ndarray,
        w_obs: np.ndarray,
        x_pred: np.ndarray,
        w_pred: np.ndarray,
    ) -> plt.Figure:
        """Plot predicted vs observed displacement."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_pred, w_pred, "b-", linewidth=2, label="PINN prediction")
        ax.plot(x_obs, w_obs, "ro", markersize=6, label="Observations")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("w (mm)")
        ax.set_title("Predicted vs Observed Displacement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Moment-curvature
    # ------------------------------------------------------------------

    def plot_moment_curvature(
        self,
        kappa: np.ndarray,
        M: np.ndarray,
    ) -> plt.Figure:
        """Plot moment-curvature response."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(kappa, M, "b-", linewidth=2)
        ax.set_xlabel("Curvature (1/mm)")
        ax.set_ylabel("Moment (N.mm)")
        ax.set_title("Moment-Curvature Response")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
