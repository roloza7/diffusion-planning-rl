from math import prod
from typing import Any, Callable, Optional
import torch
from torch import (
    Tensor,
    LongTensor
)
import torch.nn as nn
import torch.nn.functional as F
from dprl.algo.utils import cosine_noise_schedule, extract
from einops import rearrange
import warnings
from dprl.algo.models.transformer import Transformer
from hydra.utils import instantiate
from torch.distributions import Independent, OneHotCategorical

class Diffusion(nn.Module):
    def __init__(self,
                 *,
                 x_shape : torch.Size,
                 x_external_cond_dim : int,
                 timesteps : int = 1000,
                 sampling_timesteps : int = 200,
                 schedule_fn : str = 'cosine',
                 schedule_fn_kwargs : dict[str, Any] = {},
                 is_causal : bool = True,
                 clip_noise : float = 20.0,
                 snr_clip : float = 5.0,
                 stabilization_level : float = 10,
                 ddim_sampling_eta : float = 1.0,
                 objective : str = "pred_noise",
                 noise_type : str = "normal", # normal | categorical
                 categorical_dim : int = 16,
                 model_kwargs : dict[str, Any] = {},
                 ):
        super().__init__()
        self.x_shape = x_shape
        self.x_external_cond_dim = x_external_cond_dim
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.schedule_fn = schedule_fn
        self.schedule_fn_kwargs = schedule_fn_kwargs
        self.is_causal = is_causal
        self.clip_noise = clip_noise
        self.snr_clip = snr_clip
        self.stabilization_level = stabilization_level
        self.ddim_sampling_eta = ddim_sampling_eta
        self.objective = objective
        self.noise_type = noise_type
        self.categorical_dim = categorical_dim
        self._build_buffers()
        self._build_model(**model_kwargs)
    
    def _build_model(self, hidden_size, num_layers, nhead, dim_feedforward):
        
        x_dim = prod(self.x_shape)
        self.flatten = len(self.x_shape) > 1
        
        if self.flatten:
            warnings.warn(f"x_shape contains more than one dimension. Dimensions will be flattened by default but you might want to add your own behavior before passing to this function")

        transformer = Transformer(
            x_dim=hidden_size,
            external_cond_dim=self.x_external_cond_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )   
        
        # TODO: Add all these hyperparameters to a hydra conf somewhere
        self.model = transformer
        self.x_shape = (x_dim,)
    
    def _build_buffers(self):
        if self.schedule_fn == 'cosine':
            beta_schedule_fn = cosine_noise_schedule
        else:
            raise ValueError(f"Unknown beta schedule {self.schedule_fn}")
        
        # basic diffusion parameters
        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)
        
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        
        if self.sampling_timesteps > self.timesteps:
            raise ValueError(f"Sampling timesteps cannot be larger than training timesteps ({self.sampling_timesteps} > {self.timesteps})")
        
        """
        As proposed in https://arxiv.org/pdf/2010.02502
        """
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps
        
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alpha_bars", alpha_bars.to(torch.float32))
        self.register_buffer("alpha_bars_prev", alpha_bars_prev.to(torch.float32))

        # Simplifying forward and backward diffusion calculations
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars).to(torch.float32))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars).to(torch.float32))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars).to(torch.float32))
        self.register_buffer("sqrt_recip_minus_one_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1).to(torch.float32))
        
        """
        Deriving loss weights according to a min-snr weighing strategy
        As Proposed in https://arxiv.org/pdf/2303.09556
        """
        signal_to_noise_ratio = alpha_bars / (1 - alpha_bars)
        clipped_snr = torch.clamp(signal_to_noise_ratio, max=self.snr_clip)
        
        self.register_buffer("clipped_snr", clipped_snr)
        self.register_buffer("snr", signal_to_noise_ratio)
        

    def forward_sample(self, x_start : Tensor, t : LongTensor, noise : Tensor = None) -> Tensor:
        """
        Performs a forward sample of the variable x at time t
        
        Args:
            x_start (Tensor): variable to sample at times t (B, T, *)
            t (LongTensor): timesteps (B, T)
            noise (Tensor): noise to add to sampled variables (B, T, *)
            
        Returns:
            x_noisy (Tensor): q(x_t|x_0) for times t (B, T, *)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
            
        return extract(self.sqrt_alpha_bars, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape) * noise
    
    def predict_start_from_noise(self, x_t : Tensor, t : LongTensor, noise : Tensor) -> Tensor:
        """
        Predicts x_0 given model noise outputs
        """
        return extract(self.sqrt_recip_alpha_bars, t, x_t.shape) * x_t - extract(self.sqrt_recip_minus_one_alpha_bars, t, x_t.shape) * noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alpha_bars, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recip_minus_one_alpha_bars, t, x_t.shape)
    
    def model_predictions(self, x_t : Tensor, t : LongTensor, external_cond : Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Gets model noise predictions
        
        Args: 
            x_t (Tensor): noisy samples from q
            t (Tensor): timesteps of each sample
            external_cond (Tensor): external conditioning variable
            
        Return:
            epsilon (Tensor): noise prediction from the model
            x_start (Tensor): clipped model's prediciton of p(x_0|x_t)
            model_output (Tensor): unclipped model noise prediction
        """
        model_output = self.model(x_t, t, external_cond, is_causal=self.is_causal)
        
        """
        Baseline reference implementation
        https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
        implements multiple objectives, but https://arxiv.org/pdf/2006.11239 finds noise prediction to be more stable, so we only use that
        """
        if self.objective == "pred_noise":
            epsilon = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x_t, t, epsilon)
            
        elif self.objective == "pred_x0":
            x_start = model_output
            epsilon = self.predict_noise_from_start(x_t, t, x_start)
        
        return (epsilon, x_start, model_output)
    
    def compute_loss_weights(self, noise_levels : LongTensor) -> Tensor:
        """
        Computes loss weights via min-snr weighing strategy
        https://arxiv.org/pdf/2303.09556

        Args:
            noise_levels (LongTensor): noise levels (B, T) 
        """
        
        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        # norm_clipped_snr = clipped_snr / self.snr_clip
        # norm_snr = snr / self.snr_clip
        
        if self.objective == "pred_noise":
            return clipped_snr / snr
        elif self.objective == "pred_x0":
            return clipped_snr
        
        # TODO: implement fused snr
        
        return clipped_snr / snr
        
        
    def forward(self, x : Tensor, noise_levels : LongTensor, external_cond : Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Forward step
        
        Args:
            x (Tensor): tensor containing unnoised samples (B, T, *)
            noise_levels (LongTensor): tensor containing desired noise levels (B, T)
            external_cond (Tensor): external conditioning variables (B, T, x_external_cond_dim)
        Returns:
            x_pred (Tensor): model's prediction of x_0
            loss (Tensor): prediction loss
        """
        
        x_orig_shape = x.shape
        if self.flatten:
            x = torch.flatten(x, start_dim=2)
        
        if self.noise_type == "normal":    
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        elif self.noise_type == "categorical":
            noise = torch.zeros_like(x).fill_(1.0 / self.categorical_dim)
        
        noised_x = self.forward_sample(x, noise_levels, noise=noise)
        epsilon, x_pred, pred = self.model_predictions(noised_x, noise_levels, external_cond=external_cond)
        
        if self.objective == "pred_noise":
            if self.noise_type == "categorical":
                raise ValueError("Not sure how to implement categorical noise prediction")
            target = noise
        elif self.objective == "pred_x0":
            num_states = int(x.shape[-1] // self.categorical_dim)
            pred = rearrange(pred, "b t (c k) -> b t c k", c=num_states, k=self.categorical_dim)
            pred_grad = F.softmax(pred, dim=-1)
            pred = Independent(OneHotCategorical(logits=pred), reinterpreted_batch_ndims=1).sample() + pred_grad - pred_grad.detach()
            pred = rearrange(pred, "b t c k -> b t (c k)")
            target = x
        
        
        loss = F.mse_loss(pred, target, reduction="none")
        loss_weight = self.compute_loss_weights(noise_levels)
        loss_weight = loss_weight.view(loss_weight.shape + (1,) * (loss.ndim - 2))
        loss = loss * loss_weight
        
        return x_pred.view(x_orig_shape), loss        
        
    def backward_sample(self,
                        x : Tensor,
                        external_cond : Optional[Tensor],
                        curr_noise_level : LongTensor,
                        next_noise_level : LongTensor,
                        guidance_fn: Optional[Callable] = None) -> Tensor:
        """
        Samples p(x_t-1|x_t) for one step
        Uses ddim sampling as proposed in https://arxiv.org/pdf/2010.02502 to speed up inference
        """
        
        orig_x_shape = x.shape
        if self.flatten:
            x = torch.flatten(x, start_dim=2)
        
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1, device=x.device).long()
        
        curr_noise_level = real_steps[curr_noise_level]
        next_noise_level = real_steps[next_noise_level]
        
        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x,
                external_cond,
                curr_noise_level,
                next_noise_level,
                guidance_fn
            ).view(orig_x_shape)
        
    def ddim_sample_step(self,
                         x : Tensor,
                         external_cond : Optional[Tensor],
                         curr_noise_level : Tensor,
                         next_noise_level : Tensor,
                         guidance_fn: Optional[Callable] = None) -> Tensor:
        """
        Performs a sample step based on the DDIM paper
        
        Note: I know it looks complicated, but we NEED this for this backbone to be borderline useful becasue
        we don't want to do 1000+ diffusion steps per environment action.
        This method allows us to train using a large amount of diffusion steps but then do inference with a fraction of the time
        """
        
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level
        )      
        
        orig_x = x.clone().detach()
        scaled_context = self.forward_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x)
        )
        
        add_shape_channels = lambda t: rearrange(t, f"... -> ...{' 1' * len(orig_x.shape[2:])}")
        x = torch.where(add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        alpha = self.alpha_bars[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alpha_bars[next_noise_level]
        )
        
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        )
        c = torch.sqrt(1 - alpha_next - sigma**2)
        
        alpha_next = add_shape_channels(alpha_next)
        c = add_shape_channels(c)
        sigma = add_shape_channels(sigma)
        
        if guidance_fn is not None:
            # TODO: Implement guidance. Might prove vital to the project
            # Since we have 'classifiers' in the form of models that predict reward, (action, discount, done?),
            # classifier-based guidance was kind of neglected because classifier-free was developed,
            # but in our case we need to train the prediction models anyway, so we can take advantage of that and
            # nudge the prediction towards states that are beneficial (i.e.: higher reward, cause a specific action, end the episode asap)
            #
            # See https://arxiv.org/pdf/2105.05233 sections 4.* for classifier-guidance
            # Also https://arxiv.org/pdf/2207.12598 for classifier-free guidance
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                pred_noise, x_start, _ = self.model_predictions(
                    x_t=x,
                    t=clipped_curr_noise_level,
                    external_cond=external_cond,
                )

                guidance_loss = guidance_fn(x_start)
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]

                pred_noise = pred_noise + (1 - alpha_next).sqrt() * grad
                x_start = self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise)
        else:
            pred_noise, x_start, _ = self.model_predictions(
                x_t=x,
                t=clipped_curr_noise_level,
                external_cond=external_cond
            )
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = x_start * torch.sqrt(alpha_next) + pred_noise * c + sigma * noise
        
        # only update frames where snr increases
        mask = curr_noise_level == next_noise_level
        
        x_pred = torch.where(
            add_shape_channels(mask),
            orig_x,
            x_pred
        )  
                
        return x_pred
        
        
if __name__ == "__main__":
    
    diffusion_model = Diffusion(
        x_shape=(3, 8, 8),
        x_external_cond_dim=2,
        timesteps=100,
        sampling_timesteps=20,
        schedule_fn='cosine',
    )
    
    batch_size, seq_len, shape = 16, 10, (3, 8, 8)
    x = torch.randn((batch_size, seq_len) + shape)
    ext_cond = torch.randn((batch_size, seq_len, 2))
    noise_levels = torch.randint(0, 100, (batch_size, seq_len))
    
    x_pred, loss = diffusion_model.forward(x, noise_levels, ext_cond)
    
    assert x_pred.shape == (batch_size, seq_len, 3, 8, 8)