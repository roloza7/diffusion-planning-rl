import torch.nn as nn
import torch.nn.functional as F
from torch import (
    Tensor
)
from einops import rearrange

class KernelDiscriminator(nn.Module):
    def __init__(self,
                 base_channels: int,
                 base_shape: tuple[int, ...],
                 patch_size: tuple[int, int],
                 stride: tuple[int, int],
                 mlp_feedforward_dim: int,
                 mlp_depth: int):
        super().__init__()
        
        self.base_channels = base_channels
        self.base_shape = base_shape
        self.patch_size = patch_size
        self.stride = stride
        self.mlp_feedforward_dim = mlp_feedforward_dim
        self.mlp_depth = mlp_depth
        
        self.emb = nn.Conv2d(base_shape[0], base_channels, kernel_size=patch_size, stride=self.stride)
        self.emb_act = nn.SELU()
        
        self.mlp_in = nn.Conv2d(base_channels, mlp_feedforward_dim, kernel_size=1, stride=1)
        self.mlp_act = nn.SELU()
        self.mlp_out = nn.Conv2d(mlp_feedforward_dim, 1, kernel_size=1, stride=1)

        self.emb = nn.utils.spectral_norm(self.emb)
        self.mlp_in = nn.utils.spectral_norm(self.mlp_in)
        self.mlp_out = nn.utils.spectral_norm(self.mlp_out)
        
    def compute_loss_weights(self, x : Tensor, output_size: tuple[int, int]):
                
        B, C, H, W = x.shape
        
        strided = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        std = torch.std(strided, dim=1, keepdim=True)
        weights = F.fold(std, output_size=output_size, kernel_size=1, stride=1)
        return weights + 0.1
               
    def forward(self, x : Tensor, need_loss_weights : bool = False):
        
        assert len(x.shape) == 5, "Tensor must be of shape (B, T, C, H, W)"
        
        B, T = x.shape[:2]
        
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        h = self.emb(x)
        h = self.emb_act(h)
        
        h = self.mlp_in(h)
        h = self.mlp_act(h)
        h = self.mlp_out(h)
        
        if not need_loss_weights:
            return h
        
        loss_weights = self.compute_loss_weights(x, h.shape[-2:])
        
        return h, loss_weights
    
if __name__ == "__main__":
    import torch
    
    D = KernelDiscriminator(
        base_channels=128,
        base_shape=(3, 64, 64),
        patch_size=(8, 8),
        stride=(2, 2),
        mlp_feedforward_dim=256,
        mlp_depth=2
    )
    
    x = torch.randn((5, 10, 3, 64, 64))
    y = D(x)
    
    print(y.shape)