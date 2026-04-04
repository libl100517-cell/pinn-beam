"""Abstract base class for constitutive material models."""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseMaterial(ABC):
    """Unified interface for constitutive models.

    All materials expose stress(strain), tangent(strain), and parameter
    get/set methods.  Subclasses implement the specific constitutive law.
    """

    @abstractmethod
    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress from strain (vectorised)."""

    @abstractmethod
    def tangent(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute tangent modulus dσ/dε (vectorised)."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Return current material parameters as a plain dict."""

    @abstractmethod
    def set_parameters(self, **kwargs: float) -> None:
        """Update material parameters by keyword."""
