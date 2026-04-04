"""Generic multi-layer perceptron building block."""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected MLP with configurable depth, width, and activation.

    Parameters
    ----------
    in_dim : int
        Input dimension (typically 1 for x or ξ).
    out_dim : int
        Output dimension (typically 1 for a single field).
    hidden_dims : list of int
        Width of each hidden layer.
    activation : str
        Activation function name: "tanh", "relu", "gelu", "sin".
    """

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        hidden_dims: List[int] | None = None,
        activation: str = "tanh",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 32, 32]

        act_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "sin": SinActivation,
        }
        act_cls = act_map.get(activation, nn.Tanh)

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinActivation(nn.Module):
    """Sine activation (useful for periodic / smooth function approximation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
