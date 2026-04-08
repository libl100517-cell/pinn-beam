"""Core training loop for PINN beam models."""

import os
import logging
from typing import Dict, List

import torch

from models.pinn_beam import PINNBeamModel
from utils.logger import TrainingLogger
from utils.ntk_weights import compute_ntk_weights, compute_gradnorm_weights
from utils.sampling import residual_resample


class Trainer:
    """Runs the PINN training loop.

    Parameters
    ----------
    model : PINNBeamModel
    optimizer : torch.optim.Optimizer
    scheduler : optional learning rate scheduler.
    log_dir : str, optional
        Directory for training log file. If None, no file logging.
    use_ntk : bool
        Enable NTK-based adaptive loss weighting.
    ntk_every : int
        Recompute NTK weights every this many epochs.
    ntk_alpha : float
        EMA smoothing factor for NTK weights.
    resample_every : int
        Resample collocation points every this many epochs. 0 = disabled.
    """

    def __init__(
        self,
        model: PINNBeamModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        log_dir: str | None = None,
        use_ntk: bool = False,
        ntk_every: int = 100,
        ntk_alpha: float = 0.1,
        resample_every: int = 0,
        ntk_max_ratio: float = 100.0,
        use_gradnorm: bool = False,
        gradnorm_every: int = 100,
        gradnorm_alpha: float = 0.05,
        warmup_keys: list | None = None,
        warmup_epochs: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = TrainingLogger()
        self.use_ntk = use_ntk
        self.ntk_every = ntk_every
        self.ntk_alpha = ntk_alpha
        self.resample_every = resample_every
        self.ntk_max_ratio = ntk_max_ratio
        self.use_gradnorm = use_gradnorm
        self.gradnorm_every = gradnorm_every
        self.gradnorm_alpha = gradnorm_alpha
        self._ntk_weights: Dict[str, float] | None = None
        self._gradnorm_ema: Dict[str, float] | None = None
        self.warmup_keys = warmup_keys or []
        self.warmup_epochs = warmup_epochs

        # File logger
        self._file_logger = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._file_logger = logging.getLogger("pinn_train")
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.handlers.clear()
            fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._file_logger.addHandler(fh)

    def _log(self, msg: str) -> None:
        print(msg)
        if self._file_logger is not None:
            self._file_logger.info(msg)

    def train(
        self,
        xi_col: torch.Tensor,
        xi_bc: torch.Tensor,
        q_bar: float,
        n_epochs: int,
        xi_data: torch.Tensor | None = None,
        w_data: torch.Tensor | None = None,
        print_every: int = 500,
        snapshot_fn=None,
        snapshot_every: int = 0,
        lbfgs_after: int = 0,
        lbfgs_epochs: int = 500,
        save_dir: str | None = None,
    ) -> TrainingLogger:
        """Run the training loop. Returns the TrainingLogger.

        Parameters
        ----------
        lbfgs_after : int
            Switch to L-BFGS after this many Adam epochs. 0 = disabled.
        lbfgs_epochs : int
            Number of L-BFGS iterations after switching.
        save_dir : str or None
            Directory to save best/last model weights.
        """
        params = list(self.model.field_nets.parameters())
        n_col = xi_col.shape[0]
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()

            # Weight warmup: ramp specified keys from 0 to 1 over warmup_epochs
            warmup_w = None
            if self.warmup_keys and self.warmup_epochs > 0:
                ramp = min(epoch / self.warmup_epochs, 1.0)
                warmup_w = {k: ramp for k in self.warmup_keys}

            # Merge adaptive + warmup weights
            adaptive_w = self._ntk_weights
            if warmup_w:
                adaptive_w = dict(adaptive_w) if adaptive_w else {}
                for k, v in warmup_w.items():
                    adaptive_w[k] = adaptive_w.get(k, 1.0) * v

            loss, components, raw_losses, ptw_res = self.model.forward(
                xi_col, xi_bc, q_bar,
                xi_data=xi_data, w_data=w_data,
                adaptive_weights=adaptive_w,
            )

            # NTK/GradNorm: only after warmup completes
            adaptive_ready = (epoch > self.warmup_epochs) if self.warmup_epochs > 0 else True

            if adaptive_ready and self.use_ntk and epoch % self.ntk_every == 1:
                self._ntk_weights = compute_ntk_weights(
                    raw_losses, params,
                    ema_weights=self._ntk_weights,
                    alpha=self.ntk_alpha,
                    max_ratio=self.ntk_max_ratio,
                )

            if adaptive_ready and self.use_gradnorm and epoch % self.gradnorm_every == 1:
                self._ntk_weights, self._gradnorm_ema = compute_gradnorm_weights(
                    raw_losses, params,
                    ema_log_weights=self._gradnorm_ema,
                    alpha=self.gradnorm_alpha,
                )

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Residual-based adaptive resampling
            if self.resample_every > 0 and epoch % self.resample_every == 0:
                xi_col = residual_resample(
                    n_col, ptw_res, xi_col,
                    x_min=0.0, x_max=1.0,
                    uniform_ratio=0.5,
                )

            # Logging
            self.logger.log_loss(loss.item(), components)
            # Learning rate
            self.logger.lr_history.append(self.optimizer.param_groups[0]["lr"])
            # Effective weights (manual × warmup × adaptive)
            eff_w = {}
            for name in raw_losses:
                w_manual = self.model.loss_fn.weights.get(name, 1.0)
                w_adapt = adaptive_w.get(name, 1.0) if adaptive_w else 1.0
                eff_w[name] = w_manual * w_adapt
            # Add equil_M if separate
            if "equil_M" not in eff_w and "equil_M" in components:
                eff_w["equil_M"] = self.model.loss_fn.weights.get("equil_M", 1.0)
            for name, val in eff_w.items():
                if name not in self.logger.effective_weight_history:
                    self.logger.effective_weight_history[name] = []
                self.logger.effective_weight_history[name].append(val)
            if (self.use_ntk or self.use_gradnorm) and self._ntk_weights:
                self.logger.log_ntk_weights(self._ntk_weights)
            if self.model.inverse_registry is not None:
                self.logger.log_params(self.model.inverse_registry.get_values())

            if epoch % print_every == 0 or epoch == 1:
                msg = f"Epoch {epoch:>6d} | loss = {loss.item():.6e}"
                for k, v in components.items():
                    if k != "total":
                        msg += f" | {k}={v:.3e}"
                if self.model.inverse_registry is not None:
                    for k, v in self.model.inverse_registry.get_values().items():
                        msg += f" | {k}={v:.1f}"
                if (self.use_ntk or self.use_gradnorm) and self._ntk_weights:
                    ntk_str = " ".join(f"{k}={v:.2f}" for k, v in self._ntk_weights.items())
                    msg += f" | NTK[{ntk_str}]"
                self._log(msg)

            # Snapshot callback
            if snapshot_fn and snapshot_every > 0 and epoch % snapshot_every == 0:
                snapshot_fn(epoch)

            # Track best model
            cur_loss = loss.item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_state = {k: v.clone() for k, v in
                              self.model.field_nets.state_dict().items()}

        # === L-BFGS fine-tuning phase ===
        if lbfgs_after > 0 and lbfgs_epochs > 0:
            self._log(f"\n  Switching to L-BFGS for {lbfgs_epochs} iterations...")
            lbfgs_opt = torch.optim.LBFGS(
                self.model.field_nets.parameters(),
                lr=1.0, max_iter=20, history_size=50,
                line_search_fn="strong_wolfe",
            )

            # Use full weights (warmup complete)
            final_w = {}
            for k in self.warmup_keys:
                final_w[k] = 1.0

            for lbfgs_ep in range(1, lbfgs_epochs + 1):
                def closure():
                    lbfgs_opt.zero_grad()
                    loss_l, _, _, _ = self.model.forward(
                        xi_col, xi_bc, q_bar,
                        xi_data=xi_data, w_data=w_data,
                        adaptive_weights=final_w if final_w else None,
                    )
                    loss_l.backward()
                    return loss_l

                lbfgs_opt.step(closure)

                # Evaluate for logging
                loss_eval, comp_eval, _, _ = self.model.forward(
                    xi_col, xi_bc, q_bar,
                    xi_data=xi_data, w_data=w_data,
                    adaptive_weights=final_w if final_w else None,
                )
                self.logger.log_loss(loss_eval.item(), comp_eval)
                self.logger.lr_history.append(1.0)

                if loss_eval.item() < best_loss:
                    best_loss = loss_eval.item()
                    best_state = {k: v.clone() for k, v in
                                  self.model.field_nets.state_dict().items()}

                if lbfgs_ep % 100 == 0 or lbfgs_ep == 1:
                    self._log(f"  L-BFGS {lbfgs_ep:>5d} | loss = {loss_eval.item():.6e}")

                if snapshot_fn and snapshot_every > 0 and lbfgs_ep % snapshot_every == 0:
                    snapshot_fn(n_epochs + lbfgs_ep)

        # === Save weights ===
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.field_nets.state_dict(),
                       os.path.join(save_dir, "last.pt"))
            if best_state is not None:
                torch.save(best_state, os.path.join(save_dir, "best.pt"))
            self._log(f"  Saved best (loss={best_loss:.6e}) and last weights.")

        return self.logger
