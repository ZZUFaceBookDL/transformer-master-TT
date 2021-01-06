from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.utils import generate_local_map_mask


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = False):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._mask = mask

        # Query, keys and value matrices
        # 输入维度与输出维度是对一个样本说的
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        Q = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)  # (shape batchsize * head_num , q)
        K = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)  # (shape batchsize * head_num , k)
        V = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)  # (shape batchsize * head_num , v)

        scores = torch.matmul(Q, K.transpose(-1, -2))

        if self._mask:
            mask = torch.ones_like(scores)
            mask = torch.tril(mask, diagonal=0)
            scores = torch.where(mask > 0, scores, torch.Tensor([-2**32+1]).expand_as(scores[0]))
            scores = self.dropout(scores)

        # Apply sotfmax
        scores = F.softmax(scores, dim=-1)

        attention = torch.matmul(scores, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self._W_o(attention_heads)

        return self_attention

