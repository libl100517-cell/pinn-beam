"""Configuration for simply supported beam — elastic forward analysis."""

from .base_config import BeamConfig


def get_config() -> BeamConfig:
    return BeamConfig(
        mode="forward",
        elastic=True,
        n_epochs=20000,
        learning_rate=1e-4,
        n_collocation=200,
        activation="tanh",
        N_applied=10000.0,  # 10 kN
    )
