"""Training logger for recording loss history and parameter convergence."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrainingLogger:
    """Records training metrics over epochs."""

    loss_history: List[float] = field(default_factory=list)
    component_history: Dict[str, List[float]] = field(default_factory=dict)
    param_history: Dict[str, List[float]] = field(default_factory=dict)
    ntk_weight_history: Dict[str, List[float]] = field(default_factory=dict)

    def log_loss(self, total_loss: float, components: Dict[str, float] | None = None) -> None:
        """Record total loss and optional component breakdown."""
        self.loss_history.append(total_loss)
        if components:
            for name, value in components.items():
                if name not in self.component_history:
                    self.component_history[name] = []
                self.component_history[name].append(value)

    def log_params(self, params: Dict[str, float]) -> None:
        """Record current parameter values (for inverse analysis tracking)."""
        for name, value in params.items():
            if name not in self.param_history:
                self.param_history[name] = []
            self.param_history[name].append(value)

    def log_ntk_weights(self, weights: Dict[str, float]) -> None:
        """Record NTK adaptive weights."""
        for name, value in weights.items():
            if name not in self.ntk_weight_history:
                self.ntk_weight_history[name] = []
            self.ntk_weight_history[name].append(value)
