"""Configuration for simply supported beam — inverse parameter identification."""

from .base_config import BeamConfig


def get_config() -> BeamConfig:
    cfg = BeamConfig(
        mode="inverse",
        elastic=True,
        n_epochs=15000,
        learning_rate=1e-3,
        n_collocation=80,
    )
    # moment_disp = M̄ - (EI/EI_ref)·κ̄ = 0  — couples w'' to Ec via EI
    cfg.loss_weights["moment_disp"] = 100.0
    cfg.loss_weights["const_M"] = 10.0
    cfg.loss_weights["const_N"] = 10.0
    cfg.loss_weights["data_disp"] = 100.0
    cfg.loss_weights["bc"] = 100.0
    return cfg
