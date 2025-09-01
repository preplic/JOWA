# CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=39500 src/fine_tune.py hydra/job_logging=disabled hydra/hydra_logging=disabled common.env=YarsRevenge training.action.use_imagination=True training.action.planning_horizon=2
import warnings
import os
import wandb
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from dataset import AtariTrajInMemory, collate_fn
from replay_buffer import ReplayBuffer
from train import Trainer
from utils import hydra_main

warnings.filterwarnings("ignore")

# login W&B
api_key = os.environ.get("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)

class Tuner(Trainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        
        # re-define dataset for finetune
        self.train_dataset = AtariTrajInMemory(
            data_dir='dataset/finetune/expert/trajectory/data', 
            csv_dir='dataset/finetune/expert/segment/csv', 
            envs=cfg.common.envs, 
            csv_suffix="_tau_1_right_padding", 
        )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank,
            seed=cfg.common.seed,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.training.world.batch_size // self.world_size,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.train.num_of_workers // self.local_world_size,
            sampler=self.train_sampler,
            pin_memory=True,
            prefetch_factor=8,
            drop_last=True,
        )
        
        if self.use_imagination:
            self.imagine_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                seed=cfg.common.seed + 10000,
            )
            self.imagine_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=cfg.training.world.batch_size // self.world_size,
                collate_fn=collate_fn,
                num_workers=cfg.datasets.train.num_of_workers // self.local_world_size,
                sampler=self.imagine_sampler,
                pin_memory=True,
                prefetch_factor=2,
                drop_last=True,
            )
            self.imagine_replay = ReplayBuffer(
                sequence_length=cfg.common.sequence_length, 
                capacity=len(self.train_dataset),
                obs_shape=(1, 84, 84),
                device=self.device,
            )
        
        dist.barrier()


@hydra_main(config_path="../config", config_name="finetune_150M")
def get_hydra_config(cfg: DictConfig) -> DictConfig:
    return cfg


if __name__ == '__main__':
    config = get_hydra_config()
    trainer = Tuner(config)
    trainer.run()