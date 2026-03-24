"""
A2C Agent for 2048 using Stable-Baselines3.

A2C (Advantage Actor-Critic) is an on-policy algorithm that combines
policy gradient methods with a value function baseline. It's synchronous
and simpler than PPO, making it a good baseline for comparison.

References:
    Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


# ─── Custom Feature Extractor ────────────────────────────────────────

class Game2048CNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 16-channel 2048 observations.

    Same architecture as DQN/PPO for fair comparison:
        Conv2D(16, 128, 2×2) → Conv2D(128, 128, 2×2) → Conv2D(128, 128, 2×2)
        → Flatten → FC(128, 256)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ─── Metrics Callback ────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Callback to log episode metrics during A2C training."""

    def __init__(
        self,
        logger_obj: TrainingLogger,
        eval_freq: int = 10_000,
        checkpoint_freq: int = 50_000,
        log_dir: str = "logs/a2c",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.logger_obj = logger_obj
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_dir = log_dir
        self.episode_num = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_num += 1
                metrics = EpisodeMetrics(
                    episode=self.episode_num,
                    total_score=info.get("score", info["episode"]["r"]),
                    max_tile=info.get("max_tile", 0),
                    num_moves=info["episode"]["l"],
                    valid_moves=info["episode"]["l"],
                    invalid_moves=0,
                    wall_clock_seconds=info["episode"]["t"],
                    training_steps=self.num_timesteps,
                )
                self.logger_obj.log_episode(metrics)

        if self.num_timesteps % self.eval_freq == 0 and self.verbose:
            summary = self.logger_obj.get_summary(last_n=50)
            if summary:
                tqdm.write(
                    f"  Step {self.num_timesteps:>7,} | Ep {self.episode_num:>5} | "
                    f"Avg Score: {summary['avg_score']:>7.0f} | "
                    f"Max Tile: {summary['max_tile_ever']:>5} | "
                    f"512+: {summary['reach_512']:.1%}"
                )

        if self.num_timesteps % self.checkpoint_freq == 0:
            path = os.path.join(self.log_dir, f"a2c_step_{self.num_timesteps}")
            self.model.save(path)
            self.logger_obj.plot_training_curves()

        return True


# ─── Training Function ───────────────────────────────────────────────

def train_a2c(
    total_steps: int = 1_000_000,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    log_dir: str = "logs/a2c",
    reward_mode: str = "score_delta",
    seed: int = 42,
    lr: float = 7e-4,
    n_steps: int = 5,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    n_envs: int = 1,
) -> A2C:
    """
    Train an A2C agent on 2048 using Stable-Baselines3.

    Returns:
        Trained A2C model.
    """
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(
        Gym2048Env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"reward_mode": reward_mode},
    )

    policy_kwargs = {
        "features_extractor_class": Game2048CNN,
        "features_extractor_kwargs": {"features_dim": 256},
    }

    model = A2C(
        "CnnPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device=device,
    )

    logger = TrainingLogger(log_dir=log_dir, experiment_name="a2c")
    callback = MetricsCallback(
        logger_obj=logger,
        eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq,
        log_dir=log_dir,
        verbose=1,
    )

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  A2C Training — {total_steps:,} steps              ║")
    print(f"║  Device: {model.device}  n_envs: {n_envs}              ║")
    print(f"║  Reward: {reward_mode}                    ║")
    print(f"╚══════════════════════════════════════════╝")

    pbar = tqdm(total=total_steps, desc="A2C Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            pbar.update(self.model.n_envs)
            return True

    model.learn(
        total_timesteps=total_steps,
        callback=[callback, ProgressCallback()],
    )
    pbar.close()

    model.save(os.path.join(log_dir, "a2c_final"))
    logger.plot_training_curves()
    env.close()  # closes all sub-envs if VecEnv

    print(f"\n{'='*50}")
    print("Training complete!")
    summary = logger.get_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return model
