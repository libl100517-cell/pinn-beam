"""Tests for fiber section model."""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from materials import ManderConcrete, BilinearSteel
from sections import RCRectSection


class TestFiberSection:
    def setup_method(self):
        self.concrete = ManderConcrete(fc=30.0, Ec=25000.0, eps_co=-0.1, eps_cu=-0.2)
        self.steel = BilinearSteel(fy=400.0, Es=200000.0, b=1.0)
        self.rc = RCRectSection(
            width=300.0,
            height=500.0,
            concrete=self.concrete,
            steel=self.steel,
            n_concrete_fibers=20,
            rebar_layout=[(-200.0, 1520.0), (200.0, 804.0)],
        )

    def test_zero_strain_zero_resultants(self):
        """At zero strain and zero curvature, N and M should be near zero."""
        eps0 = torch.tensor([0.0])
        kappa = torch.tensor([0.0])
        resp = self.rc.section.response(eps0, kappa)
        assert abs(resp["N"].item()) < 1e-3
        assert abs(resp["M"].item()) < 1e-3

    def test_pure_bending_symmetric_section(self):
        """Pure curvature on approximately symmetric section produces moment."""
        eps0 = torch.tensor([0.0])
        kappa = torch.tensor([1e-6])
        resp = self.rc.section.response(eps0, kappa)
        # Should produce non-zero moment
        assert abs(resp["M"].item()) > 0.0

    def test_axial_strain_produces_axial_force(self):
        """Uniform axial strain should produce axial force."""
        eps0 = torch.tensor([0.001])
        kappa = torch.tensor([0.0])
        resp = self.rc.section.response(eps0, kappa)
        # Positive strain → positive N (tension)
        assert resp["N"].item() > 0.0

    def test_gross_area(self):
        assert abs(self.rc.gross_area - 300.0 * 500.0) < 1e-6

    def test_gross_inertia(self):
        expected = 300.0 * 500.0**3 / 12.0
        assert abs(self.rc.gross_inertia - expected) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
