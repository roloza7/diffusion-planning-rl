from torch import (
    Tensor,
    empty,
    einsum
)
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Independent, OneHotCategorical
from dprl.algo.models.encoder import Encoder

class CategoricalEncoder(nn.Module):
    def __init__(self,
                 stochastic_size : int,
                 num_states : int,
                 encoder : Encoder,
                 gradient_strategy : str = "st",
                 gumbel_temperature : float = 0.5,
                 state_embedding_size : int = None,
                 **kwargs):
        super().__init__()
        
        assert stochastic_size * num_states == encoder.out_size, "Encoder output should be equal to stochastic_size * num_states"
        self.stochastic_size = stochastic_size
        self.num_states = num_states
        self.gradient_strategy = gradient_strategy
        self.gumbel_temperature = gumbel_temperature # only needed for gumbel strategy
        self.encoder = encoder
        self.state_embedding_size = state_embedding_size or stochastic_size
        self.out_size = self.state_embedding_size * self.num_states
        self.codebook = nn.Parameter(
            empty((self.num_states, self.stochastic_size, self.state_embedding_size))
        )
        
        self._init_codebook()
        
    def _init_codebook(self):
        nn.init.xavier_normal_(self.codebook.data)
    
    def forward(self, x : Tensor, needs_grad : bool = True):
        
        N, S = self.num_states, self.stochastic_size
        
        h = self.encoder(x)
        
        logits = rearrange(h, "b t (n s) -> b t n s", n=N, s=S)
        
        dist = Independent(OneHotCategorical(logits=logits, validate_args=True),
                           reinterpreted_batch_ndims=1)
        
        h_sample = dist.sample() # b t n s
        
        if needs_grad:
            if self.gradient_strategy == "st":
                probs = F.softmax(logits, dim=-1)
                h_sample = h_sample + probs - probs.detach()
            elif self.gradient_strategy == "gumbel":
                soft_probs = F.gumbel_softmax(logits, tau=self.gumbel_temperature)
                h_sample = h_sample + soft_probs - soft_probs.detach()
            else:
                raise ValueError("Invalid gradient strategy")
        
        hidden = einsum("btck,cke->btce", h_sample, self.codebook)
        
        hidden = rearrange(hidden, "b t n e -> b t (n e)")
        
        return hidden
        
        
        
if __name__ == "__main__":
    import torch

    encoder = Encoder(base_channels=16,
                      in_shape=(3, 64, 64),
                      channels_mult=(1, 1, 1, 1),
                      out_size=64,
                      mlp_feedforward_dim=128)
    
    encoder = CategoricalEncoder(
        stochastic_size=8,
        num_states=8,
        encoder=encoder,
        gradient_strategy="st"
    )
    
    B, T = 16, 20
    C, H, W = 3, 64, 64
    
    x = torch.randn((B, T, C, H, W))
    y = encoder(x)
    
    
    