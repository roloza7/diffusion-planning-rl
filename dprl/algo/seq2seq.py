from typing import Optional, Self
from lightning import Fabric
import numpy as np
from omegaconf import DictConfig
import torch
from torch import (
    Tensor,
)
import torch.nn as nn
from dprl.algo.utils import generate_noise_levels
from dprl.algo.models import (
    Diffusion,
    Decoder,
    ActionModel
)
from dprl.algo.categorical import CategoricalEncoder
from einops import rearrange
import tqdm
from torch.distributions import Independent, OneHotCategorical

import torch.nn.functional as F 

class LatentDFModel(nn.Module):
    def __init__(self,
                 diffusion : nn.Module,
                 encoder : CategoricalEncoder,
                 decoder : Decoder,
                 action_model : ActionModel,
                 *,
                 uncertainty_scale : float,
                 chunk_size : int,
                 sliding_window_ctx : int,
                 train_autoencoder : bool):
        super().__init__()
        self.diffusion : Diffusion = diffusion
        self.encoder = encoder
        self.decoder = decoder
        self.action_model = action_model
        self.train_autoencoder = train_autoencoder
        
        # TODO: Add hydra config
        self.uncertainty_scale = uncertainty_scale
        self.chunk_size = chunk_size
        self.sliding_window_ctx = sliding_window_ctx
                
    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    def __call__(self, obs : Tensor, external_cond : Tensor = None, act : Tensor = None):
        """
        Performs a forward step of the latent diffusion model
        
        Args:
            obs (Tensor): observation (B, N, *)
            external_cond (Tensor): external conditioning for diffusion (B, N, external_cond_dim) 
        """
        
        info = {}
        
        # G Training step
        z, encoder_info = self.encoder(obs) #B, T, E
        info = info | encoder_info
                
        z_states = rearrange(z, "b t (c k) -> b t c k", k=self.diffusion.categorical_dim)
        z_states = (z_states + 0.1) / (self.diffusion.categorical_dim * 0.1 + 1.0)
        z_dist = Independent(OneHotCategorical(probs=z_states, validate_args=True), reinterpreted_batch_ndims=1)
        
        # (B, L, -1)        
        noise_levels = generate_noise_levels(obs, masks=None, max_noise_level=self.diffusion.timesteps)
                
        z_pred, diffusion_loss = self.diffusion.forward(z, noise_levels, external_cond, base_dist=z_dist)
        
        info["diffusion/loss"] = diffusion_loss.mean()
        
         
        x_pred = self.decoder(z_pred)
        
        if self.train_autoencoder:
            # TODO: KL div is probably better here, more justifiable on report
            # diffusion loss is still going down on libero 90 after the AE converges more or less
            ae_loss = F.mse_loss(self.decoder(z_pred), obs)
            info["loss/autoencoder"] = ae_loss
                        
        if act != None:
            B, T, E = z_pred.shape

            z_pred = torch.as_strided(z_pred, size=(B, T - 1, E * 2), stride=(T * E, E, 1))
            action_predictions : torch.distributions.Normal = self.action_model(z_pred)
            info["action_entropy"] = action_predictions.entropy()
            action_loss = -action_predictions.log_prob(act).mean()
            info["loss/action"] = action_loss
        
        return x_pred, action_loss + diffusion_loss.mean() + ae_loss, info
    
    @torch.inference_mode()
    def block_sample(self,
                     obs : Tensor,
                     n_frames : int,
                     mask : torch.BoolTensor,
                     *,
                     external_cond : Optional[Tensor] = None,
                     need_frames = False,
                     ) -> Tensor:
        batch_size, n_frames = obs.shape[:2]
        scheduling_matrix = self._generate_fullsequence_scheduling_matrix(n_frames, self.uncertainty_scale, self.diffusion.sampling_timesteps, mask.cpu())
        
        z, _ = self.encoder(obs)
        
        z = z.clone()
        
        
        
        noise = torch.zeros_like(z).fill_(1.0 / self.diffusion.categorical_dim)
        z = torch.where(
            ~mask[None, :, None].cuda(),
            z,
            noise
        )
        
        frames = []
        
        for m in tqdm.trange(scheduling_matrix.shape[0]- 1):
            from_noise_levels = scheduling_matrix[None, m].repeat(batch_size, axis=0)
            to_noise_levles = scheduling_matrix[None, m + 1].repeat(batch_size, axis=0)
            
            z = self.diffusion.backward_sample(
                z,
                external_cond=external_cond,
                curr_noise_level=from_noise_levels,
                next_noise_level=to_noise_levles
            )
            
            if need_frames:
                frames.append(z.clone())
            
        x_pred = self.decoder(z)
            
        if need_frames:    
            return x_pred, frames
        return x_pred
        
  
    @torch.inference_mode()
    def sample(self,
               obs : Tensor,
               n_frames : int,
               *,
               external_cond : Optional[Tensor] = None
               ) -> Tensor:
        
        
        batch_size, n_context_frames = obs.shape[:2]
        curr_frame : int = 0
        
        # Context Frames
        x_pred = obs.clone()
        curr_frame += n_context_frames
        
        frames = []
        
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            scheduling_matrix = self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale, self.diffusion.sampling_timesteps)
            chunk = torch.randn((batch_size, horizon) + x_pred.shape[2:], device=obs.device)
            chunk = torch.clamp(chunk, -self.diffusion.clip_noise, self.diffusion.clip_noise)
            x_pred = torch.cat([x_pred, chunk], dim=1)
            
            # Sliding windows
            start_frame = max(0, curr_frame + horizon - 99)
            # TODO: Implement guidance function for the end-to-end model
            
            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[None, :].repeat(batch_size, axis=0)
                to_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
                )[None, :].repeat(batch_size, axis=0)
                
                # print(from_noise_levels)
                # print(to_noise_levels)
                x_pred[:, start_frame:] = self.diffusion.backward_sample(
                    x=x_pred[:, start_frame:],
                    external_cond=external_cond[:, start_frame : curr_frame + horizon] if external_cond is not None else None,
                    curr_noise_level=from_noise_levels[:, start_frame:],
                    next_noise_level=to_noise_levels[:, start_frame:]
                )
                
                frames.append(x_pred.clone())
            
            curr_frame += horizon
                
        return x_pred, frames
    
    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float, sampling_timesteps : int):
        height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, sampling_timesteps)
    
    def _generate_fullsequence_scheduling_matrix(self, horizon : int, uncertainty_scale: float, sampling_timesteps : int, mask : torch.BoolTensor):
        height = sampling_timesteps
        scheduling_matrix = np.zeros((height + 1, horizon), dtype=np.int64)
        mask = mask.numpy()
        for m in range(height):
            scheduling_matrix[m] = height - m
        
        scheduling_matrix[:, mask] = 0
            
        return np.clip(scheduling_matrix, 0, sampling_timesteps)
    
    @staticmethod
    def from_config(fabric : Fabric, cfg : DictConfig, encoder : CategoricalEncoder, decoder : Decoder, action_model : ActionModel, train_autoencoder : bool = False) -> Self:
                                
        diffusion = Diffusion(
            x_shape=(cfg.encoder.out_size,),
            **cfg.diffusion_model
        )
        
        model = LatentDFModel(
            diffusion=diffusion,
            encoder=encoder,
            decoder=decoder,
            action_model=action_model,
            **cfg.latent_df_model,
            train_autoencoder=train_autoencoder
        )
        
        if train_autoencoder == False:
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False
            for param in model.action_model.parameters():
                param.requires_grad = False
        
        return fabric.setup_module(model) 