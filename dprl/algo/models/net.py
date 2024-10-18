import torch.nn as nn
import torch.nn.functional as F
from torch import (
    Tensor
)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip_in = nn.InstanceNorm2d(out_channels)
        
    def forward(self, x : Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.in1(h)
        h = F.selu(h)
        h = self.conv2(h)
        h = self.in2(h)        
        return F.selu(self.skip_in(self.skip(x)) + h)
    
class MLP(nn.Module):
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 feedfoward_dim : int):
        super().__init__()
        
        self.aff1 = nn.Linear(in_features, feedfoward_dim)
        self.aff2 = nn.Linear(feedfoward_dim, out_features)
        
    def forward(self, x):
        x = F.selu(self.aff1(x))
        return self.aff2(x)
        