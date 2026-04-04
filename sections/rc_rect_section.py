"""Rectangular reinforced concrete section builder.

Creates a FiberCollection from geometric inputs: width, height, concrete
material, steel bars with positions and areas.
"""

from typing import List, Tuple

from materials.base_material import BaseMaterial
from .fibers import Fiber, FiberCollection
from .fiber_section import FiberSection


class RCRectSection:
    """Builder for a rectangular RC fiber section.

    Parameters
    ----------
    width : float
        Section width (mm).
    height : float
        Section total height (mm).
    concrete : BaseMaterial
        Concrete constitutive model.
    steel : BaseMaterial
        Steel constitutive model.
    n_concrete_fibers : int
        Number of concrete fibers along section height.
    rebar_layout : list of (y, area)
        Each entry is (distance from centroid in mm, bar area in mm^2).
    """

    def __init__(
        self,
        width: float,
        height: float,
        concrete: BaseMaterial,
        steel: BaseMaterial,
        n_concrete_fibers: int = 20,
        rebar_layout: List[Tuple[float, float]] | None = None,
    ):
        self.width = width
        self.height = height
        self.concrete = concrete
        self.steel = steel
        self.n_concrete_fibers = n_concrete_fibers
        self.rebar_layout = rebar_layout or []

        self._section = self._build()

    def _build(self) -> FiberSection:
        """Discretise the section into fibers."""
        fibers = FiberCollection()
        h = self.height
        fiber_h = h / self.n_concrete_fibers

        # Concrete fibers – centroid at y = 0
        for i in range(self.n_concrete_fibers):
            y_bot = -h / 2.0 + i * fiber_h
            y_center = y_bot + fiber_h / 2.0
            area = self.width * fiber_h
            fibers.add(Fiber(y=y_center, area=area, material=self.concrete))

        # Steel reinforcement fibers
        for y_bar, A_bar in self.rebar_layout:
            fibers.add(Fiber(y=y_bar, area=A_bar, material=self.steel))

        return FiberSection(fibers)

    @property
    def section(self) -> FiberSection:
        """Return the assembled FiberSection."""
        return self._section

    @property
    def gross_area(self) -> float:
        return self.width * self.height

    @property
    def gross_inertia(self) -> float:
        """Second moment of area about the centroid (mm^4)."""
        return self.width * self.height ** 3 / 12.0
