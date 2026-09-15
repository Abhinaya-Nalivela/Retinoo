'''Temperature scaling for model calibration.

A simple wrapper around a learnable scalar ``temperature``. The forward
method divides the logits by the temperature value. This matches the
behaviour expected by the pipeline tests.
''' 

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialise temperature close to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero – clamp to a tiny epsilon
        temp = torch.clamp(self.temperature, min=1e-6)
        return logits / temp
