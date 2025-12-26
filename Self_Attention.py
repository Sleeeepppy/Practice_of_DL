# Attention(Q, K, V) = softmax((QK^T) / (sqrt(d_k))V
import math
from turtle import forward
import torch
import torch.nn as nn

class Self_Attentnion(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.softmax = nn.Softmax()

    def forward(self, X):
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        attention_value = torch.matmul(q, k.transpose(-2, -1))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)
        # 针对最后一个维度softmax
        output = torch.matmul(attention_weight, v)
        print(output)
        return output
        
X = torch.rand(3, 2, 4)
att = Self_Attentnion(X.shape[2])
att(X)