"""Trainable inverse parameter registry.

Provides a clean mechanism for registering constitutive parameters as
trainable torch Parameters, with optional physical bounds (enforced via
soft clamping).

Usage
-----
    registry = InverseParameterRegistry()
    registry.register("Ec", init_value=25000.0, bounds=(10000.0, 50000.0))
    registry.register("fc", init_value=30.0, bounds=(15.0, 60.0))

    # Add registry.parameters() to the optimiser alongside the PINN nets
    optimiser = torch.optim.Adam(
        list(field_nets.parameters()) + list(registry.parameters()),
        lr=1e-3,
    )

    # During training, retrieve the current (clamped) value:
    Ec = registry.get("Ec")
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class InverseParameterRegistry(nn.Module):
    """Registry for trainable material parameters.

    Each registered parameter is stored as an unconstrained nn.Parameter.
    When retrieved via ``get()``, optional bounds are enforced using a
    sigmoid mapping to [lo, hi].
    """

    def __init__(self):
        super().__init__()
        self._params: Dict[str, nn.Parameter] = {}
        self._bounds: Dict[str, Tuple[float, float] | None] = {}

    def register(
        self,
        name: str,
        init_value: float,
        bounds: Tuple[float, float] | None = None,
        trainable: bool = True,
    ) -> None:
        """Register a parameter.

        Parameters
        ----------
        name : str
            Unique parameter name (e.g. "Ec").
        init_value : float
            Initial dimensional value.
        bounds : (lo, hi) or None
            Physical bounds.  If provided, the parameter is mapped through
            sigmoid so its effective value stays in (lo, hi).
        trainable : bool
            If False the parameter is fixed.
        """
        if bounds is not None:
            lo, hi = bounds
            # Inverse sigmoid to initialise the raw parameter
            frac = (init_value - lo) / (hi - lo)
            frac = max(min(frac, 0.999), 0.001)
            raw = torch.log(torch.tensor(frac / (1.0 - frac)))
        else:
            raw = torch.tensor(float(init_value))

        param = nn.Parameter(raw, requires_grad=trainable)
        self._params[name] = param
        self._bounds[name] = bounds
        # Register with nn.Module so it appears in self.parameters()
        self.register_parameter(name, param)

    def get(self, name: str) -> torch.Tensor:
        """Return the effective (bounded) value as a differentiable tensor."""
        raw = self._params[name]
        bounds = self._bounds[name]
        if bounds is not None:
            lo, hi = bounds
            return lo + (hi - lo) * torch.sigmoid(raw)
        return raw

    def get_dict(self) -> Dict[str, torch.Tensor]:
        """Return all parameters as a {name: value} dict."""
        return {name: self.get(name) for name in self._params}

    def get_values(self) -> Dict[str, float]:
        """Return all parameters as plain floats (detached)."""
        return {name: self.get(name).item() for name in self._params}
