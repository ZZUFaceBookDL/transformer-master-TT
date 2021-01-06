import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.multiHeadAttention import MultiHeadAttention
from tst.positionwiseFeedForward import PositionwiseFeedForward


class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 dropout: float = 0.3,
                 mask: bool = False):
        """Initialize the Decoder block"""
        super().__init__()

        MHA = MultiHeadAttention
        self._selfAttention = MHA(d_model, q, v, h, mask)
        self._encoderDecoderAttention = MHA(d_model, q, v, h, mask)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Encoder-decoder attention
        residual = x
        x = self._encoderDecoderAttention(query=x, key=memory, value=memory)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm3(x + residual)

        return x
