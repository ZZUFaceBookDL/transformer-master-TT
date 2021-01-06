import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.multiHeadAttention import MultiHeadAttention
from tst.positionwiseFeedForward import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 dropout: float = 0.3,
                 mask: bool = False):
        """Initialize the Encoder block"""
        super().__init__()

        MHA = MultiHeadAttention
        self._selfAttention = MHA(d_model, q, v, h, mask)
        self._feedForward = PositionwiseFeedForward(d_model)

        # 归一化层 不过不是将数据限定在 0-1 之间或者 高斯分布（正态分布）上
        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        # Dropout 源码就是调用了 function中的dropout函数
        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        # 输入x为 encoder之前进行的embedding的输出  维度：（batchsize, dmodel）
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
