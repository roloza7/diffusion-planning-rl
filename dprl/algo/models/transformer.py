import math
import torch
import torch.nn as nn
from torch import (
    Tensor,
    LongTensor
)
from einops import rearrange
from itertools import chain

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Pos Emb
    Snippet taken from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    to mimic the approach in the "Forced Diffusion" paper
    """
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AttnBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 nhead,
                 dim_feedforward,
                 dropout):
        super().__init__()
    
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout,
        )
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.ln1 = nn.RMSNorm(hidden_size)
        self.ln2 = nn.RMSNorm(hidden_size)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, hidden_size),
        )
        
    def forward(self, x, mask, is_causal):
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        
        x = self.ln1(x + self.mha(q, k, v)[0])
        x = self.ln2(x + self.ff(x))
        
        return x
    


class Transformer(nn.Module):
    """
    Transformer module designed to fit into diffusion block
    
    Adapted from the implementation from https://arxiv.org/pdf/2407.01392
    
    Functioning:
        
                                    ┌─────┐   ┌────┐   ┌─────┐
    Input-------------------------> │     │-->│    │-->│     │
                    ┌─────────────┐ │     │-->│    │-->│     │
    Timesteps------>│Sin. Pos. Emb│>│ MLP │-->│ TF │-->│ MLP │
                    └─────────────┘ │ IN  │-->│    │-->│ OUT │
                    ┌─────────────┐ │     │-->│    │-->│     │
    Conditioning--->│Sin. Pos. Emb│>│     │-->│    │-->│     │
                    └─────────────┘ │     │-->│    │-->│     │
                                    └─────┘   └────┘   └─────┘
                                    
    Predictions are processed in (B, L, )
    """
    
    
    
    def __init__(self,
                 x_dim : int,
                 external_cond_dim : int,
                 hidden_size : int = 128,
                 num_layers : int = 4,
                 nhead : int = 4,
                 dim_feedforward : int = 512,
                 dropout : float = 0.6):
        super().__init__()
        self.external_cond_dim = external_cond_dim
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=hidden_size,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.attn_blocks = nn.ModuleList([
            AttnBlock(
                hidden_size,
                nhead,
                dim_feedforward,
                dropout
            ) for _ in range(num_layers)
        ])

        # TIMESTEP embedding
        # do not confuse with position embedding, this embeds the diffusion time
        k_embed_dim = hidden_size // 2
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        # Position embedding
        # embeds frame position in REAL time, not diffusion time
        self.t_embed = SinusoidalPosEmb(dim=hidden_size)
        self.in_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim + external_cond_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.out_mlp = nn.Linear(hidden_size, x_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        # for name, param in chain.from_iterable([self.in_mlp.named_parameters(), self.out_mlp.named_parameters(), self.transformer.named_parameters()]):
        #     if 'weight' in name and param.data.dim() == 2:
        #         nn.init.(param, 0.05)
        # for name, param in self.out_mlp.named_parameters():
        #     if 'bias'  in name:
        #         nn.init.normal_(param, 0.5)
        pass
        
        
    
    def forward(self,
                x : Tensor,
                noise_levels : LongTensor,
                external_cond : Tensor = None,
                is_causal : bool = False) -> Tensor:
        """
        This step is used in the backward diffusion step
        e_t ~ p(e_t|x_t, t, cond)
        
        Args:
            x (Tensor): noisy samples (B, T, C)
            noise_levels (LongTensor): diffusion noise levels (B, T)
            external_cond (Tensor): conditioning for generation (B, T, external_cond_dim)
            
        Returns:
            x (Tensor): epsilon ~ predicted noise value
        """
        
        batch_size, seq_len, _ = x.shape
        k_embed = rearrange(self.k_embed(noise_levels.flatten()), "(b t) d -> b t d ", t=seq_len)
        x = torch.cat([x, k_embed], dim=-1)
        if external_cond is not None:
            x = torch.cat([x, external_cond], dim=-1)
            
        # x is (B, T, x.shape + k_embed | x.shape + k_embed + external_cond_dim)
        x = self.in_mlp(x)
        x = x + rearrange(self.t_embed(torch.arange(seq_len, device=x.device)[:, None]), "t b d -> b t d")
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, x.device) if is_causal else None
        
        for block in self.attn_blocks:
            x = block(x, mask, is_causal)
        
        x = self.out_mlp(x)
        return x
    
if __name__ == "__main__":
    model = Transformer(x_dim=32,
                        external_cond_dim=2,
                        hidden_size=128,
                        num_layers=1,
                        nhead=4,
                        dim_feedforward=128)
    
    batch_size, seq_len, dim = 32, 10, 32
    x = torch.randn((batch_size, seq_len, dim))
    ext = torch.randn((batch_size, seq_len, 2))
    noise_levels = torch.randint(0, 16, (batch_size, seq_len), dtype=torch.long)
    
    print(x.shape, ext.shape, noise_levels.shape)
    epsilon = model.forward(x, noise_levels, ext, is_causal=True)
    print(epsilon.shape)
    
    assert epsilon.shape == (batch_size, seq_len, 32)
    