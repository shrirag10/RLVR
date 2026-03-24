"""
Discrete SAC Agent for 2048.

Standard SAC (Soft Actor-Critic) is designed for continuous action spaces.
This implements SAC for discrete actions following Christodoulou (2019),
using an entropy-regularized objective that encourages exploration through
a stochastic policy over the 4 discrete moves.

Key differences from continuous SAC:
    - Policy outputs a categorical distribution over actions (softmax)
    - Entropy is computed from the categorical distribution
    - No reparameterization trick needed

References:
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Max Entropy Deep RL" (2018)
    Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019)
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


# ─── Network Architecture ────────────────────────────────────────────

class DiscreteSACNetwork(nn.Module):
    """
    Shared CNN backbone with separate actor (policy) and critic (Q-value) heads.

    Architecture follows the same CNN as DQN/PPO/A2C for fair comparison.
    """

    def __init__(self, n_channels: int = 16, n_actions: int = 4):
        super().__init__()

        # Shared CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, 4, 4)
            n_flat = self.backbone(sample).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
        )

        # Actor head: outputs action logits
        self.actor = nn.Linear(256, n_actions)

        # Twin Q-networks (Double Q-learning)
        self.q1 = nn.Linear(256, n_actions)
        self.q2 = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor):
        """Returns (action_probs, q1_values, q2_values)."""
        features = self.fc(self.backbone(x))
        logits = self.actor(features)
        action_probs = F.softmax(logits, dim=-1)
        q1 = self.q1(features)
        q2 = self.q2(features)
        return action_probs, q1, q2

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        """Sample an action from the policy."""
        action_probs, _, _ = self.forward(x)
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action, action_probs


# ─── Replay Buffer ───────────────────────────────────────────────────

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)


# ─── Discrete SAC Agent ──────────────────────────────────────────────

class DiscreteSACAgent:
    """
    Soft Actor-Critic for discrete actions.

    Uses entropy-regularized RL to maintain exploration while learning
    an optimal policy. The temperature parameter α is automatically tuned.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 5000,
        target_entropy_ratio: float = 0.3,  # lowered: 0.5 caused alpha divergence on 2048
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        # Networks
        self.online = DiscreteSACNetwork().to(self.device)
        self.target = DiscreteSACNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        # Separate optimizers: backbone+fc shared, actor head, critic heads
        # FIX: do NOT include backbone in actor optimizer alone — share it via
        # a single shared_optimizer so it isn't updated twice per step.
        self.shared_optimizer = Adam(
            list(self.online.backbone.parameters()) +
            list(self.online.fc.parameters()),
            lr=lr,
        )
        self.actor_optimizer = Adam(
            list(self.online.actor.parameters()),
            lr=lr,
        )
        self.critic_optimizer = Adam(
            list(self.online.q1.parameters()) +
            list(self.online.q2.parameters()),
            lr=lr,
        )

        # Automatic entropy tuning
        n_actions = 4
        self.target_entropy = -np.log(1.0 / n_actions) * target_entropy_ratio
        # FIX: initialise log_alpha to log(0.1) ≈ -2.3 instead of 0 (=1.0),
        # and clamp after every update to prevent exponential explosion.
        self.log_alpha = torch.tensor([np.log(0.1)], requires_grad=True,
                                      dtype=torch.float32, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr * 0.1)  # slower alpha LR
        self.LOG_ALPHA_MIN = -10.0   # alpha ≥ exp(-10) ≈ 4.5e-5
        self.LOG_ALPHA_MAX =  2.0    # alpha ≤ exp(2)  ≈ 7.4

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(
        self, state: np.ndarray, valid_actions: list[int] | None = None,
        deterministic: bool = False
    ) -> int:
        """Select action using the stochastic policy."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _, _ = self.online(state_t)

            if valid_actions is not None:
                # Mask invalid actions
                mask = torch.zeros(4, device=self.device)
                for a in valid_actions:
                    mask[a] = 1.0
                action_probs = action_probs * mask
                action_probs = action_probs / (action_probs.sum() + 1e-8)

            # NaN guard: fall back to uniform over valid actions if probs are bad
            if torch.isnan(action_probs).any() or action_probs.sum() < 1e-6:
                if valid_actions:
                    return valid_actions[np.random.randint(len(valid_actions))]
                return np.random.randint(4)

            if deterministic:
                action = action_probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

        return action

    def update(self) -> dict | None:
        """Perform one gradient step."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ─── Critic Update ────────────────────────────────────
        with torch.no_grad():
            next_probs, next_q1, next_q2 = self.target(next_states)
            next_q = torch.min(next_q1, next_q2)
            # V(s') = Σ π(a|s') * [Q(s',a) - α * log π(a|s')]
            next_log_probs = torch.log(next_probs + 1e-8)
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        _, q1, q2 = self.online(states)
        q1_selected = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_selected = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q1_selected, target_q) + F.mse_loss(q2_selected, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.online.q1.parameters()) + list(self.online.q2.parameters()), 1.0
        )
        self.critic_optimizer.step()

        # ─── Actor Update ─────────────────────────────────────
        action_probs, q1_detach, q2_detach = self.online(states)
        q_min = torch.min(q1_detach.detach(), q2_detach.detach())
        log_probs = torch.log(action_probs + 1e-8)
        # Actor loss: minimize α*H - Q
        actor_loss = (action_probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()

        self.shared_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.online.backbone.parameters()) +
            list(self.online.fc.parameters()) +
            list(self.online.actor.parameters()), 1.0
        )
        self.shared_optimizer.step()
        self.actor_optimizer.step()

        # ─── Alpha (Temperature) Update ───────────────────────
        entropy = -(action_probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # FIX: clamp log_alpha to prevent exponential divergence to NaN
        with torch.no_grad():
            self.log_alpha.clamp_(self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)

        # ─── Soft Target Update ───────────────────────────────
        for param, target_param in zip(self.online.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "entropy": entropy.item(),
        }

    def save(self, path: str):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "log_alpha": self.log_alpha.data,
        }, path + ".pt")

    def load(self, path: str):
        """Load agent checkpoint."""
        if not path.endswith(".pt"):
            path = path + ".pt"
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data = checkpoint["log_alpha"].to(self.device)


# ─── Training Loop ────────────────────────────────────────────────────

def train_sac(
    total_steps: int = 500_000,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    log_dir: str = "logs/sac",
    reward_mode: str = "score_delta",
    seed: int = 42,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    learning_starts: int = 5000,
    device: str = "auto",
) -> DiscreteSACAgent:
    """
    Train a Discrete SAC agent on 2048.

    Returns:
        Trained DiscreteSACAgent.
    """
    os.makedirs(log_dir, exist_ok=True)

    env = Gym2048Env(reward_mode=reward_mode, seed=seed)
    agent = DiscreteSACAgent(
        lr=lr, gamma=gamma, tau=tau,
        buffer_size=buffer_size, batch_size=batch_size,
        learning_starts=learning_starts, device=device,
    )

    logger = TrainingLogger(log_dir=log_dir, experiment_name="sac")

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Discrete SAC Training — {total_steps:,} steps     ║")
    print(f"║  Device: {agent.device}                          ║")
    print(f"║  Reward: {reward_mode}                    ║")
    print(f"╚══════════════════════════════════════════╝")

    obs, info = env.reset()
    episode_num = 0
    episode_start = time.time()
    episode_moves = 0

    pbar = tqdm(range(1, total_steps + 1), desc="SAC Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for step in pbar:
        # Select action
        valid_actions = info.get("valid_actions", list(range(4)))
        action = agent.select_action(obs, valid_actions, deterministic=False)

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_moves += 1

        # Store transition
        agent.buffer.push(Transition(obs, action, reward, next_obs, done))

        # Update
        if step >= learning_starts:
            agent.update()

        obs = next_obs

        if done:
            episode_num += 1
            metrics = EpisodeMetrics(
                episode=episode_num,
                total_score=info.get("score", 0),
                max_tile=info.get("max_tile", 0),
                num_moves=episode_moves,
                valid_moves=episode_moves,
                invalid_moves=0,
                wall_clock_seconds=time.time() - episode_start,
                training_steps=step,
            )
            logger.log_episode(metrics)

            obs, info = env.reset()
            episode_start = time.time()
            episode_moves = 0

        # Periodic reporting
        if step % eval_freq == 0:
            summary = logger.get_summary(last_n=50)
            if summary:
                tqdm.write(
                    f"  Step {step:>7,} | Ep {episode_num:>5} | "
                    f"Avg Score: {summary['avg_score']:>7.0f} | "
                    f"Max Tile: {summary['max_tile_ever']:>5} | "
                    f"α: {agent.alpha:.3f}"
                )

        # Checkpointing
        if step % checkpoint_freq == 0:
            agent.save(os.path.join(log_dir, f"sac_step_{step}"))
            logger.plot_training_curves()

    # Final save
    agent.save(os.path.join(log_dir, "sac_final"))
    logger.plot_training_curves()
    env.close()

    print(f"\n{'='*50}")
    print("Training complete!")
    summary = logger.get_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return agent
