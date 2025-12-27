# Attention(Q, K, V) = softmax((QK^T) / (sqrt(d_k))V
# add some details: dropout, attention mask and output matrix

import math
from turtle import forward
import torch
import torch.nn as nn

class Self_Attentnion_v3(nn.Module):
    def __init__(self, hidden_dim: int = 728, dropout_rate = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout_rate)
        # optional
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.softmax = nn.Softmax()

    def forward(self, X, attention_mask=None):
        qkv = self.qkv_proj(X)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)
        attention_value = torch.matmul(q, k.transpose(-2, -1))
        attention_weight = attention_value / math.sqrt(self.hidden_dim)
        # print("attention mask:", attention_weight)
        
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e20"))
            # print("attention weight:", attention_weight)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)
        output = torch.matmul(attention_weight, v)
        output = self.output_proj(output)
        # print("output:", output)
        return output
        
X = torch.rand(3, 4, 2)
mask = torch.tensor(
    [[1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0]])
# print("shape of mask:", mask.shape)
mask = torch.unsqueeze(mask, dim=1).repeat(1, 4, 1)
# print("shape of repeated mask:", mask.shape)
att = Self_Attentnion_v3(X.shape[2])
att(X, mask)