"""Fiber primitives for section discretisation.

A Fiber stores the position (distance from section centroid), tributary area,
and a reference to its constitutive material.
"""

from dataclasses import dataclass
from typing import List

import torch

from materials.base_material import BaseMaterial


@dataclass
class Fiber:
    """Single fiber in a section.

    Attributes
    ----------
    y : float
        Distance from section centroid (positive upward).
    area : float
        Tributary area of the fiber.
    material : BaseMaterial
        Constitutive model assigned to this fiber.
    """
    y: float
    area: float
    material: BaseMaterial


class FiberCollection:
    """Ordered collection of fibers forming a cross-section discretisation."""

    def __init__(self, fibers: List[Fiber] | None = None):
        self.fibers: List[Fiber] = fibers or []

    def add(self, fiber: Fiber) -> None:
        self.fibers.append(fiber)

    def positions(self, device: torch.device | None = None) -> torch.Tensor:
        """Fiber centroid positions as (n_fibers,) tensor."""
        return torch.tensor([f.y for f in self.fibers], dtype=torch.float32, device=device)

    def areas(self, device: torch.device | None = None) -> torch.Tensor:
        """Fiber areas as (n_fibers,) tensor."""
        return torch.tensor([f.area for f in self.fibers], dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.fibers)
