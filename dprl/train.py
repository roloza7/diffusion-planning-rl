

import glob
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch.utils
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities import AttributeDict
from wandb.integration.lightning.fabric import WandbLogger
import wandb
import hydra
from einops import rearrange

from dprl.data.datasets import LiberoDatasetAdapter
from dprl.metrics import grad_norm
from dprl.data.utils import collate_fn, AtariTransform
from dprl.algo.autoencoder import CategoricalAutoEncoder


def train(cfg : DictConfig) -> None:
    print("Train function invoked")
    torch.autograd.set_detect_anomaly(False)
    # Might have to remove if you're using an old gpu
    # Will affect training speed but not an issue - older gpus use the equivalent of 'highest'
    torch.set_float32_matmul_precision('high') 
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckp_dir = os.path.join(output_dir, "checkpoints/")
    print(f"output_dir={output_dir}")
    
    wandb_logger = WandbLogger(log_model="all", save_dir=output_dir, project="diffusion-planing-rl")
    fabric = L.Fabric(accelerator='auto', devices=cfg.algo.n_devices, num_nodes=cfg.algo.n_nodes, precision="bf16-mixed", loggers=[wandb_logger])
    fabric.launch()
    
    if fabric.global_rank == 0:
        if os.path.isdir(ckp_dir) == False:
            os.mkdir(ckp_dir)
    
    model : CategoricalAutoEncoder
    model, generator_optim, discriminator_optim = CategoricalAutoEncoder.from_config(fabric, cfg.algo)
    model.mark_forward_method(model.generator_step)
    model.mark_forward_method(model.discriminator_step)
        
    # Compilation increases speeds by 20-35%, but only really works on linux with triton installed
    # Blame openai for rejecting PR's that add windows support
    if cfg.algo.compile == True:
        model = torch.compile(model)
    
    
    
    # Initializing breakout dataset
    # TODO: hydra config for this
    transform = AtariTransform(
        to_size=cfg.algo.encoder.in_shape[1:],
        swap_channels=True,
        num_channels=3
    )

    dataset = LiberoDatasetAdapter("datasets/libero_spatial",
                                 slice_len=50,
                                 transform=transform,
                                 frameskip=1
                                 )
        
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.algo.batch_size,
        pin_memory=True,
        num_workers=0 if cfg.dry_run else 8,
        persistent_workers=None if cfg.dry_run else True,
        prefetch_factor=None if cfg.dry_run else 2,
        drop_last=False,
        sampler=None,
        collate_fn=collate_fn
    )

    dataloader = fabric.setup_dataloaders(dataloader)
    
    # Start training loop
    global_step = 0
    
    state = AttributeDict(model=model,
                          generator_optim=generator_optim,
                          discriminator_optim=discriminator_optim,
                          global_step=global_step)
    
    checkpoints = glob.glob(f'outputs/{cfg.algo.name}/**/checkpoints/*.ckp')
    if len(checkpoints) > 0:
        checkpoint = max(checkpoints, key=os.path.getctime)
        fabric.load(checkpoint, state=state)
    use_gan = cfg.algo.categorical.loss_strategy == "gan"

    print("Starting training loop")
    while global_step < cfg.algo.total_steps:
        for batch in dataloader:
            obs = batch["observations"]
            act = batch["actions"][:, :-1]

            # TODO : Implement action guidance here too
            
            """
            Discriminator optimization step
            """
            if use_gan:
                discriminator_optim.zero_grad(set_to_none=True)
                d_grad, d_info = model.discriminator_step(obs)
                fabric.backward(d_grad)
                d_info["grad_norm"] = grad_norm(model.discriminator, fabric.device)
                discriminator_optim.step()
            
            """
            Generator optimization step
            """
            generator_optim.zero_grad(set_to_none=True)
            g_loss, g_reconstruction, g_info = model.generator_step(obs, act=act)
            fabric.backward(g_loss)
            g_info["encoder/grad_norm"] = grad_norm(model.encoder, fabric.device)
            g_info["decoder/grad_norm"] = grad_norm(model.decoder, fabric.device)
            generator_optim.step()
            
            # Logging
            if cfg.dry_run == False:
                log = {}
                if use_gan:
                    for key, value in d_info.items():
                        log[f"train/discriminator/{key}"] = value
                for key, value in g_info.items():
                    log[f"train/generator/{key}"] = value
                log[f"trainer/global_step"] = state.global_step
                if state.global_step % cfg.metrics.media_every == 0:
                    nrow = g_reconstruction.shape[1]
                    image_grid = make_grid(rearrange(g_reconstruction, "b t c h w -> (b t) c h w"), nrow=nrow)
                    image = wandb.Image(image_grid.permute(1, 2, 0).cpu().numpy(), caption=f"Reconstructions at time {state.global_step}")
                    log["train/reconstructions"] = image
                fabric.log_dict(log)

            
            # eval_metrics = eval(fabric, eval_dataloader, model)
            # log = log | eval_metrics
           
            # Long-term Metrics
            if cfg.dry_run == False and state.global_step % cfg.metrics.save_every == cfg.metrics.save_every - 1:
                fabric.save(os.path.join(ckp_dir, f"checkpoint_{state.global_step}.ckp"), state)
                
            state.global_step += 1