import math
from typing import Optional
from omegaconf import DictConfig
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from einops import parse_shape, rearrange

def generate_noise_levels(xs : torch.Tensor, masks : Optional[torch.Tensor] = None, max_noise_level : int = 4000) -> torch.Tensor:
        # TODO : Add masking
        batch_size, num_frames = xs.shape[:2]
        noise_levels = torch.randint(0, max_noise_level, (batch_size, num_frames), device=xs.device)
        return noise_levels

def extract(onedim : Tensor, a : LongTensor, shape : tuple[int, ...]) -> Tensor:
    """
    extracts values from one-dimensional tensor into multiple dimensions
    
    Args:
        onedim (Tensor): 1-dimensional tensor to extract values from
        a (LongTensor): n-dimensional tensor to index values in onedim
        shape (tuple[int, ...]): target size (will broadcast dims > n)
    
    Returns:
        out (Tensor): tensor containing values from onedim indexed by a, broadcasted to shape
    """
    base_shape = a.shape
    out = onedim[a]
    return out.reshape((base_shape) + (1,) * (len(shape) - len(base_shape)))
    
def cosine_noise_schedule(timesteps, s=0.008):
    """
    constructs a sinusoidal noise schedule
    as proposed in https://arxiv.org/pdf/2102.09672.
    
    code snippet adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py.
    """
    # 1001 timesteps because we will take sequential ratios
    f_t = torch.linspace(0, 1, steps=timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((f_t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

"""
Extremely useful class-and-function pair from the forced-diffusion implementation
https://github.com/buoyancy99/diffusion-forcing/blob/e55bc47a5ab48742156648e51d919df7725202d1/algorithms/diffusion_forcing/models/utils.py#L77
"""
class EinopsWrapper(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        return x

def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule

    
def catvae_from_config(cfg : DictConfig) -> nn.Module:
    """
    stub
    """
    return None

if __name__ == "__main__":
    betas = cosine_noise_schedule(1000)
    print(betas.shape)
    print(betas)