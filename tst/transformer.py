import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
import torch.nn.functional as F


import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_channel: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 mask: bool = False,
                 pe: bool = False,
                 noise: float = 0):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_input = d_input
        self._d_model = d_model
        self._d_channel = d_channel
        self.pe = pe
        self._noise = noise

        self.layers_encoding_1 = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask=mask) for _ in range(N)])

        self.layers_encoding_2 = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask=mask) for _ in range(N)])

        self._embedding_input = nn.Linear(d_input, d_model)
        self._embedding_channel = nn.Linear(d_channel, d_model)


        self.gate = nn.Linear(d_model * d_channel + d_model * d_input, 2)
        self.linear = nn.Linear(d_model * d_channel + d_model * d_input, d_output)
        self.layer_normal = torch.nn.LayerNorm(d_output)

        self.batch_normal = torch.nn.BatchNorm1d(d_model * d_channel + d_model * d_input)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = x.unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], self._d_channel)

        if self._noise:
            mask = torch.rand_like(x[0])  # 平均分布填充

            temp = x
            temp[:, :-1] = temp[:, 1:]

            # temp1 = x[:, 1:]
            # temp2 = x[:, -1:]
            # temp = torch.cat([temp1, temp2], dim=1)

            # x = torch.where(mask <= self._noise, torch.Tensor([0]).expand_as(x[0]), x)  # 随机变为0

            x = torch.where(mask <= self._noise, temp, x)  # 随机变为时间步+1的值


        # socre = input
        encoding_1 = self._embedding_channel(x)  # 降维 13 -> 512

        # 位置编码 ------------------------------------- 自己写的
        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch    .Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        # Encoding stack
        for layer in self.layers_encoding_1:
            encoding_1 = layer(encoding_1)

        # score = channel
        encoding_2 = x.transpose(-1, -2)
        encoding_2 = self._embedding_input(encoding_2)
        if self.pe:
            pe = torch.ones_like(encoding_2[0])
            position = torch.arange(0, self._d_channel).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_2 = encoding_2 + pe

        for layer in self.layers_encoding_2:
            encoding_2 = layer(encoding_2)

        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)

        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        output = self.layer_normal(F.relu(self.linear(encoding)))



        return output
