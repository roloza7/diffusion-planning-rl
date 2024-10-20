from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dprl.algo.models.net import ResidualBlock, MLP
from einops import rearrange
from math import prod, log2

class Decoder(nn.Module):
    def __init__(self,
                in_size : int,
                cnn_in_shape : tuple[int, ...],
                base_channels : int = 128,
                channels_mult : tuple[int, ...] = (1, 2, 4, 2),
                out_shape : tuple[int, ...] = (3, 64, 64),
                mlp_feedforward_dim : int = 128,
                norm : str = "rsm"):
        
        super().__init__()
        
        self.in_size = in_size
        self.base_channels = base_channels
        self.cnn_in_channels = cnn_in_shape[0]
        self.cnn_in_shape = cnn_in_shape[1:]
        self.channels_mult = channels_mult
        self.norm = norm
        self.channels = tuple([int(self.base_channels // mult) for mult in channels_mult])
        self.out_shape = out_shape

        self.mlp = MLP(
            in_features=self.in_size,
            out_features=prod(cnn_in_shape),
            feedfoward_dim=mlp_feedforward_dim
        )
        self.mlp_act = nn.SELU()
        
        self.demb = ResidualBlock(in_channels=self.cnn_in_channels, out_channels=self.channels[0])
        self.demb_act = nn.SELU()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(self.channels[i], self.channels[i+1]) for i in range(0, len(self.channels) - 1)
        ])
        
        self.proj = nn.Conv2d(in_channels=self.channels[-1], out_channels=out_shape[0], kernel_size=1, stride=1)
        
        if (pred_shape := Decoder.get_cnn_out_shape(self.cnn_in_shape, self.channels)[1:]) != out_shape[1:]:
            needed_blocks = log2(out_shape[1] / pred_shape[1])
            raise ValueError(f"Must have correct amount of residual blocks to upscale. You need {int(needed_blocks)} block(s)")        
        
    @staticmethod
    def get_cnn_out_shape(in_shape : tuple[int, ...], channels : tuple[int, ...]):
        H, W = in_shape[-2:]
        
        size_mult = 2 ** (len(channels) - 1) # We upscale between layers
        
        out_H, out_W = int(H * size_mult), int(W * size_mult)
        out_channels = channels[-1]
        
        return (out_channels, out_H, out_W)
        
        
    def forward(self, x : Tensor):
                
        assert len(x.shape) == 3, "Tensors should be supplied in (B, T, E) form"
        
        B, T = x.shape[:2]
        
        h = self.mlp_act(self.mlp(x))
        
        h = rearrange(h, "b t (c h w) -> (b t) c h w", c=self.cnn_in_channels, h=self.cnn_in_shape[0], w=self.cnn_in_shape[1])
                
        h = self.demb_act(self.demb(h))
        
        for i, block in enumerate(self.blocks):
            h = F.interpolate(h, scale_factor=2)
            h = block(h)
            
        out = F.tanh(self.proj(h))
        
        out = rearrange(out, "(b t) c h w -> b t c h w", b=B, t=T)
        
        return out
            
if __name__ == "__main__":
    import torch
    
    
    decoder = Decoder(in_size=32,
                      cnn_in_shape=(128, 8, 8),
                      base_channels=128,
                      channels_mult=(1, 2, 4, 2),
                      out_shape=(3, 64, 64),
                      mlp_feedforward_dim=128)
    
    B, T, E = 16, 20, 32
    
    x = torch.randn((B, T, E))
    y = decoder(x)
                            
        
        
        
        