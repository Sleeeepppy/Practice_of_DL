# Attention(Q, K, V) = softmax((QK^T) / (sqrt(d_k))V
# version for interview

from turtle import forward
import torch
import torch.nn as nn
import math

class Self_Attention_Interview(nn.Module):
    def __init__(self, hidden_dim:int=728, dropout_rate:float=0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, dim)
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        # weight shape: (batch, seq, seq)
        attention_weight = q @ k.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-inf"))
        print("attention weight:", attention_weight)
        attention_weight = torch.softmax(attention_weight, dim=-1)

        attention_weight = self.attention_dropout(attention_weight)

        output = attention_weight @ v
        print(output)
        return output

X = torch.rand(3, 4, 2)
mask = torch.tensor([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0]
])
mask = mask.unsqueeze(dim=1).repeat(1, 4, 1)
net = Self_Attention_Interview(X.shape[2])
net(X, mask)



    