from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 512):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)
        self.layer_normal = nn.LayerNorm(d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self._linear1(x)
        temp = F.relu_(temp)
        temp = self.layer_normal(temp)
        temp = self._linear2(temp)

        return temp
