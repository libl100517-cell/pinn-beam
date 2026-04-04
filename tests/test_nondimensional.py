"""Tests for non-dimensionalization consistency."""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics import NonDimScales


class TestNonDimScales:
    def setup_method(self):
        self.scales = NonDimScales(
            L=3000.0,
            E_ref=25000.0,
            A_ref=150000.0,
            I_ref=3.125e9,
        )

    def test_roundtrip_x(self):
        x = torch.tensor([1500.0])
        xi = self.scales.to_nondim_x(x)
        x_back = self.scales.to_dim_x(xi)
        assert abs(x_back.item() - 1500.0) < 1e-6

    def test_roundtrip_w(self):
        w = torch.tensor([10.0])
        w_bar = self.scales.to_nondim_w(w)
        w_back = self.scales.to_dim_w(w_bar)
        assert abs(w_back.item() - 10.0) < 1e-6

    def test_roundtrip_M(self):
        M = torch.tensor([1e8])
        M_bar = self.scales.to_nondim_M(M)
        M_back = self.scales.to_dim_M(M_bar)
        assert abs(M_back.item() - 1e8) < 1.0

    def test_roundtrip_N(self):
        N = torch.tensor([50000.0])
        N_bar = self.scales.to_nondim_N(N)
        N_back = self.scales.to_dim_N(N_bar)
        assert abs(N_back.item() - 50000.0) < 1e-3

    def test_nondim_x_at_L_is_one(self):
        x = torch.tensor([self.scales.L])
        xi = self.scales.to_nondim_x(x)
        assert abs(xi.item() - 1.0) < 1e-10

    def test_q_roundtrip(self):
        q = 20.0
        q_bar = self.scales.to_nondim_q(q)
        q_back = self.scales.to_dim_q(q_bar)
        assert abs(q_back - 20.0) < 1e-10

    def test_scale_relations(self):
        """F_ref = E_ref * A_ref, M_ref = E_ref * I_ref / L."""
        assert abs(self.scales.F_ref - 25000.0 * 150000.0) < 1e-3
        assert abs(self.scales.M_ref - 25000.0 * 3.125e9 / 3000.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
