"""Configuration for simply supported beam — elasto-plastic forward analysis."""

from .base_config import BeamConfig


def get_config() -> BeamConfig:
    return BeamConfig(
        mode="forward",
        elastic=False,
        n_epochs=10000,
        learning_rate=1e-4,
        n_collocation=200,
        activation="tanh",
        q=40.0,
        N_applied=0.0,
    )
