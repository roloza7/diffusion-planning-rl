import os
# Silence annoying tf warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
from dprl.train import train
from dprl.train_diffusion import train_diffusion
from dprl.data.utils import dotdict

@hydra.main(version_base=None, config_path="dprl/config", config_name="config")
def main(cfg : DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.DICT_CONFIG))
    
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "train_diffusion":
        train_diffusion(cfg)

    
if __name__ == "__main__":
    main()