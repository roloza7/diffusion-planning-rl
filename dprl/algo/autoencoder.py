from typing import Iterable, Self
from omegaconf import DictConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import (
    Tensor
)
from dprl.algo.categorical import CategoricalEncoder
from dprl.algo.models import Encoder, Decoder, KernelDiscriminator, ActionModel
from hydra.utils import instantiate
from lightning import Fabric
from torch.optim import Optimizer

class CategoricalAutoEncoder(nn.Module):
    def __init__(self,
                 encoder : CategoricalEncoder,
                 decoder : Decoder,
                 action_model : ActionModel,
                 discriminator : KernelDiscriminator,
                 loss_strategy : str = "mse"
                 ):
        super().__init__()
            
        self.encoder = encoder
        self.decoder = decoder
        self.action_model = action_model
        self.discriminator = discriminator
        self.loss_strategy = loss_strategy
        
    def forward(self, x : Tensor) -> Tensor:
        return self.generator_step(x)
    
    def setup_optimizers(self, cfg : DictConfig):
        generator_optim = instantiate(cfg.generator.optim, params=list(self.encoder.parameters()) + list(self.decoder.parameters()))
        if self.loss_strategy == "gan": 
            discriminator_optim = instantiate(cfg.discriminator.optim, params=self.discriminator.parameters())
            return [generator_optim, discriminator_optim]   
        return [generator_optim]
        
    
    def generator_step(self, x : Tensor, act : Tensor = None) -> Tensor:
        """
        stub
        """
        info = {}
        
        hidden = self.encoder(x)
        reconstruction = self.decoder(hidden)
        
        if self.loss_strategy == "gan":
            loss = - self.discriminator(reconstruction).mean() # Maximize fake_pred
        elif self.loss_strategy == "mse":
            loss = F.mse_loss(reconstruction, x)
        info["loss/reconstruction"] = loss
        
        if act != None:
            B, T, E = hidden.shape

            hidden = torch.as_strided(hidden, size=(B, T - 1, E * 2), stride=(B, E, 1))
            action_predictions : torch.distributions.Normal = self.action_model(hidden)
            action_loss = -action_predictions.log_prob(act).sum(dim=-1).mean()
            info["loss/action"] = action_loss
            loss += action_loss
            
        return loss, reconstruction, info
        
    def discriminator_step(self, obs : Tensor) -> Tensor:
        """
        stub
        """
        if self.loss_strategy != "gan":
            return torch.Tensor([0], device=obs.device), {}
        
        info = {}
        
        
        with torch.no_grad():
            h = self.encoder(obs)
            reconstruction = self.decoder(h)
            
        x = torch.cat([obs, reconstruction], dim=0)
        y_hat = self.discriminator(x)
        real_pred, fake_pred = torch.chunk(y_hat, chunks=2, dim=0)
        grad =  - (real_pred.mean() - fake_pred.mean()) # Maximize real_pred, minimize fake_pred
        info["grad"] = grad
        return grad, info
    
    @staticmethod
    def from_config(fabric : Fabric, cfg : DictConfig) -> tuple[Self, Optimizer, Optimizer]:
        
        encoder = Encoder(**cfg.encoder)
        cat_encoder = CategoricalEncoder(
            encoder=encoder,
            **cfg.categorical
        )
        decoder = Decoder(**cfg.decoder, cnn_in_shape=encoder.cnn_out_shape, in_size=cat_encoder.out_size)
        
        use_gan = cfg.categorical.loss_strategy == "gan"

        discriminator = KernelDiscriminator(**cfg.discriminator) if use_gan else None
        
        action_model = ActionModel(
            **cfg.action_model
        )
        
        model = CategoricalAutoEncoder(
            encoder=cat_encoder,
            decoder=decoder,
            discriminator=discriminator,
            action_model=action_model,
            loss_strategy=cfg.categorical.loss_strategy
        )
        
        if use_gan:
            model, generator_optim, discriminator_optim = fabric.setup(model, *model.setup_optimizers(cfg.categorical))
        else:
            model, generator_optim = fabric.setup(model, *model.setup_optimizers(cfg.categorical))
            discriminator_optim = None
        
        return model, generator_optim, discriminator_optim