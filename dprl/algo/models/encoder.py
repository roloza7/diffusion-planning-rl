from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dprl.algo.models.net import ResidualBlock, MLP
from einops import rearrange
from math import prod

class Encoder(nn.Module):
    def __init__(self,
                base_channels : int = 128,
                in_shape : tuple[int, ...] = (3, 64, 64),
                channels_mult : tuple[int, ...] = (1, 2, 4, 2),
                out_size : int = 64,
                mlp_feedforward_dim : int = 128,
                norm : str = "rsm"):
        
        super().__init__()
        
        self.base_channels = base_channels
        self.in_channels = in_shape[0]
        self.in_shape = in_shape[1:]
        self.channels_mult = channels_mult
        self.norm = norm
        self.channels = tuple([self.base_channels * mult for mult in channels_mult])
        self.out_size = out_size
        
        
        self.emb = ResidualBlock(in_channels=self.in_channels, out_channels=self.channels[0])
        self.emb_act = nn.SELU()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(self.channels[i], self.channels[i+1]) for i in range(0, len(self.channels) - 1)
        ])
        
        self.cnn_out_shape = Encoder.get_cnn_out_shape(self.in_shape, self.channels)
        self.cnn_out_size = prod(self.cnn_out_shape)
        
        self.mlp = MLP(
            in_features=self.cnn_out_size,
            out_features=self.out_size,
            feedfoward_dim=mlp_feedforward_dim
        )
        
    @staticmethod
    def get_cnn_out_shape(in_shape : tuple[int, ...], channels : tuple[int, ...]):
        H, W = in_shape[-2:]
        
        size_div = 2 ** (len(channels) - 1) # We avg. pool between layers
        
        assert H % size_div == 0, "Height must be divisible by number of residual blocks minus one"
        assert W % size_div == 0, "width must be divisible by number of residual blocks minus one"
        
        out_H, out_W = int(H / size_div), int(W / size_div)
        out_channels = channels[-1]
        
        return (out_channels, out_H, out_W)
        
        
    def forward(self, x : Tensor):
        
        assert len(x.shape) == 5, "Tensors should be supplied in (B, T, C, H, W) form"
        
        B, T = x.shape[:2]
        
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        emb = self.emb_act(self.emb(x))
        
        h = emb
        
        for i, block in enumerate(self.blocks):
            h = F.avg_pool2d(h, (2, 2), (2, 2))
            h = block(h)
        h = rearrange(h, "(b t) c h w -> b t (c h w)", b=B, t=T)

        return self.mlp(h)
            
if __name__ == "__main__":
    import torch
    
    
    encoder = Encoder(base_channels=16,
                      in_shape=(3, 64, 64),
                      channels_mult=(1, 1, 1, 1),
                      out_size=64,
                      mlp_feedforward_dim=128)
    
    B, T = 16, 20
    C, H, W = 3, 64, 64
    
    print(encoder.cnn_out_shape)
    x = torch.randn((B, T, C, H, W))
    y = encoder(x)
                            
        
        
        
        