"""Tests for constitutive material models."""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from materials import ManderConcrete, BilinearSteel


class TestManderConcrete:
    def test_zero_strain_zero_stress(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0)
        eps = torch.tensor([0.0])
        sig = mat.stress(eps)
        assert abs(sig.item()) < 1e-10

    def test_compression_negative_stress(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0)
        eps = torch.tensor([-0.001])
        sig = mat.stress(eps)
        assert sig.item() < 0.0, "Compressive stress should be negative"

    def test_peak_stress_near_fc(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0, eps_co=-0.002)
        eps = torch.tensor([mat.eps_co])
        sig = mat.stress(eps)
        # At peak strain, stress should be close to -fc
        assert abs(sig.item() + mat.fc) < 2.0, f"Peak stress {sig.item()} not near -fc={-mat.fc}"

    def test_tension_ascending(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0)
        eps_cr = mat.eps_cr
        eps = torch.tensor([eps_cr * 0.5])
        sig = mat.stress(eps)
        expected = mat.Ec * eps.item()
        assert abs(sig.item() - expected) < 1e-6, "Ascending branch should be σ = Ec·ε"

    def test_tension_peak_at_cracking(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0)
        eps = torch.tensor([mat.eps_cr])
        sig = mat.stress(eps)
        assert abs(sig.item() - mat.ft) < 1e-4, f"At cracking, stress should be ft={mat.ft}"

    def test_tension_softening(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0, Gf=0.1, h=150.0)
        eps_mid = (mat.eps_cr + mat.eps_tu) / 2.0
        eps = torch.tensor([eps_mid])
        sig = mat.stress(eps)
        assert 0.0 < sig.item() < mat.ft, "Softening branch should be between 0 and ft"

    def test_tension_zero_beyond_eps_tu(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0, Gf=0.1, h=150.0)
        eps = torch.tensor([mat.eps_tu * 1.5])
        sig = mat.stress(eps)
        assert abs(sig.item()) < 1e-10, "Beyond eps_tu, stress should be zero"

    def test_fracture_energy_eps_tu(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0, Gf=0.1, h=150.0)
        expected = 2.0 * 0.1 / (mat.ft * 150.0)
        assert abs(mat.eps_tu - expected) < 1e-10

    def test_crushing_zero_stress(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0, eps_cu=-0.004)
        eps = torch.tensor([-0.005])
        sig = mat.stress(eps)
        assert abs(sig.item()) < 1e-10, "Beyond crushing, stress should be zero"

    def test_get_set_parameters(self):
        mat = ManderConcrete(fc=30.0, Ec=25000.0)
        params = mat.get_parameters()
        assert params["fc"] == 30.0
        assert "Gf" in params
        mat.set_parameters(fc=40.0)
        assert mat.fc == 40.0


class TestBilinearSteel:
    def test_zero_strain_zero_stress(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0)
        eps = torch.tensor([0.0])
        sig = mat.stress(eps)
        assert abs(sig.item()) < 1e-10

    def test_elastic_range(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0)
        eps = torch.tensor([0.001])  # below yield (fy/Es = 0.002)
        sig = mat.stress(eps)
        expected = 200000.0 * 0.001
        assert abs(sig.item() - expected) < 1e-3

    def test_yield_stress(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0, b=0.01)
        eps = torch.tensor([0.003])  # beyond yield
        sig = mat.stress(eps)
        assert sig.item() > 400.0, "Post-yield stress should exceed fy"

    def test_symmetry(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0)
        eps = torch.tensor([0.001, -0.001])
        sig = mat.stress(eps)
        assert abs(sig[0].item() + sig[1].item()) < 1e-6

    def test_hardening(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0, b=0.01)
        eps_y = mat.eps_y
        eps_post = torch.tensor([eps_y + 0.01])
        sig = mat.stress(eps_post)
        expected = 400.0 + 0.01 * 200000.0 * 0.01
        assert abs(sig.item() - expected) < 1e-3

    def test_get_set_parameters(self):
        mat = BilinearSteel(fy=400.0, Es=200000.0)
        mat.set_parameters(fy=500.0)
        assert mat.fy == 500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
