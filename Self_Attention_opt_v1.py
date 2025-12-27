# Attention(Q, K, V) = softmax((QK^T) / (sqrt(d_k))V
import math
from re import split
from turtle import forward
import torch
import torch.nn as nn

class Self_Attention_opt_v1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, X):
        qkv = self.qkv_proj(X)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)
        attention_value = torch.matmul(q, k.transpose(-2, -1))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)
        # 针对最后一个维度softmax
        output = torch.matmul(attention_weight, v)
        print(output)
        return output
        
X = torch.rand(3, 2, 4)
att = Self_Attention_opt_v1(X.shape[2])
att(X)