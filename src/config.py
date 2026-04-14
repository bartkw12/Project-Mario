"""YAML configuration loader with dataclass mapping and CLI support."""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class EnvConfig:
    game: str = "SuperMarioBros-1-1-v0"
    movement: str = "SIMPLE_MOVEMENT"
    frame_skip: int = 4
    frame_stack: int = 4
    obs_size: int = 84
    num_envs: int = 8


@dataclass
class TrainingConfig:
    total_timesteps: int = 5000000
    lr: float = 0.00025
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


@dataclass
class RewardConfig:
    forward_scale: float = 0.1
    death_penalty: float = -15.0
    flag_bonus: float = 50.0


@dataclass
class EvalConfig:
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    deterministic: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            env=EnvConfig(**raw.get("env", {})),
            training=TrainingConfig(**raw.get("training", {})),
            reward=RewardConfig(**raw.get("reward", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            seed=raw.get("seed", 42),
            device=raw.get("device", "auto"),
        )


def set_global_seed(seed: int) -> None:
    """Seed torch, numpy, and stdlib random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments shared by train.py and evaluate.py."""
    parser = argparse.ArgumentParser(description="Project Mario")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 100 steps and print obs shape, then exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (evaluate.py)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video during evaluation (evaluate.py)",
    )
    return parser.parse_args()
