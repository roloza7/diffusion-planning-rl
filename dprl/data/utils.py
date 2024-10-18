from collections import defaultdict
from typing import Callable
import torch.nn as nn
import torch
from torchvision.transforms import v2
import numpy as np
from einops import rearrange
from functools import partial

class AtariTransform():
    def __init__(self,
                 to_size: tuple[int, int],
                 swap_channels : bool = False,
                 num_channels : int = 1):
        self.transforms = {
            'observations': nn.Sequential(
                v2.Resize(to_size),
                v2.RandomCrop(to_size, padding=4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5,) * num_channels, std=(0.5,) * num_channels)
            )
        }
        self.swap_channels = swap_channels
        
    def __call__(self, sample : dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in sample.items():
            out[key] = torch.from_numpy(value)
            
        if self.swap_channels:
            out['observations'] = rearrange(out['observations'], "s h w c -> s c h w")
            
        for key, transform in self.transforms.items():
            out[key] = transform(out[key])
        return out
        

def collate_fn(batch : list[dict[str, np.ndarray]]):
    data = defaultdict(list)
    
    for key in batch[0].keys():
        data[key] = torch.nn.utils.rnn.pad_sequence([obs[key] for obs in batch], batch_first=True)
    
    return data

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, d : dict):
        for key, value in d.items():
            if hasattr(value, 'keys'):
                value = dotdict(value)
            self[key] = tuple(value) if isinstance(value, list) else value