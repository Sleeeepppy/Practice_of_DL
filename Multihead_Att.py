# Multihead(Q, K, V) = Concat(head_1, head_2, ......, head_n)W^O
# head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
from turtle import forward
from sympy import Mul
import torch
import torch.nn as nn
import math

class Multihead_Att(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate:float=0.1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, attention_mask=None):
        batch, seq, dim = X.shape
        # shape(b, s, h)
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        # (b, s, h)->(b, s, head_num * head_dim)->(b, head_num, s, head_dim)
        q_state = q.view(batch, seq, self.head_num, self.head_dim).transpose(2, 1)
        k_state = k.view(batch, seq, self.head_num, self.head_dim).transpose(2, 1)
        v_state = v.view(batch, seq, self.head_num, self.head_dim).transpose(2, 1)
        # shape(b, head_nun, s, head_dim)->(b, head_num, s, s)
        attention_weight = q_state @ k_state.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-inf"))
        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)

        output_state = attention_weight @ v_state
        # shape(b, head_num, s, head_dim)->(b, s, head_num * head_dim)->(b, s, h)
        output = output_state.transpose(2, 1).reshape(batch, seq, -1)
        output = self.output_proj(output)

        return output

attention_mask = torch.tensor([
    [0, 1],
    [0, 0],
    [1, 0]
])
attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(3, 2, 2, 2)
X = torch.rand(3, 2, 8)
net = Multihead_Att(X.shape[2], 2)
net(X, attention_mask)
        







