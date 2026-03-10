import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: (batch, sequence, hidden_size)
x = torch.randn(1, 3, 64)

# QKV projections
W_q = nn.Linear(64, 32)
W_k = nn.Linear(64, 32)
W_v = nn.Linear(64, 32)

#Q = W_q(x)  # (batch, sequence, head_dim)
#K = W_k(x)  # (batch, sequence, head_dim)
#V = W_v(x)  # (batch, sequence, head_dim)

# Scaled dot-product attention
#scale = Q.shape[-1] ** 0.5
#scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (batch, sequence, sequence)
#weights = F.softmax(scores, dim=-1)                # (batch, sequence, sequence)
#out = torch.bmm(weights, V)   

# Use MultiheadAttention:
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
out,weights = mha(x, x, x)

print(out)