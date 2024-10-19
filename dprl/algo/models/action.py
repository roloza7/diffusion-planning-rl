import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class ActionModel(nn.Module):
    def __init__(self,
                 in_dim : int,
                 action_dim : int,
                 hidden_size : int,
                 depth : int):
        super().__init__()
        assert depth > 2
        
        mlp = [
            nn.Linear(in_dim * 2, hidden_size)
        ]
        for _ in range(depth - 1):
            mlp += [
                nn.SELU(),
                nn.Linear(hidden_size, hidden_size)
            ]
            
        mlp += [
            nn.SELU(),
            nn.Linear(hidden_size, action_dim)
        ]
        
        self.mlp = nn.Sequential(*mlp)
        
    def forward(self, x):
        
        mu = F.tanh(self.mlp(x))
        
        std = torch.ones_like(mu)
        
        out = Independent(Normal(mu, std), reinterpreted_batch_ndims=1)
        return out
        
                
        
            