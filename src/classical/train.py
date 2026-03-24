"""
Classical RL training and evaluation scripts.

Provides a unified CLI for training and evaluating DQN, PPO, A2C, QR-DQN, and SAC agents.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


def evaluate_agent(
    agent_type: str,
    model_path: str,
    n_episodes: int = 100,
    reward_mode: str = "score_delta",
    seed: int = 42,
    render: bool = False,
    log_dir: str = "logs/eval",
) -> dict:
    """
    Evaluate a trained agent over N episodes.

    Args:
        agent_type: 'dqn' or 'ppo'.
        model_path: Path to saved model checkpoint.
        n_episodes: Number of evaluation episodes.
        reward_mode: Reward mode for environment.
        seed: Random seed.
        render: Whether to render the game.
        log_dir: Where to save evaluation logs.

    Returns:
        Summary statistics dict.
    """
    env = Gym2048Env(
        reward_mode=reward_mode,
        seed=seed,
        render_mode="human" if render else None,
    )
    logger = TrainingLogger(
        log_dir=log_dir, experiment_name=f"{agent_type}_eval"
    )

    if agent_type == "dqn":
        from src.classical.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(model_path)
        agent.epsilon = 0.0  # Greedy evaluation

        for ep in range(1, n_episodes + 1):
            obs, info = env.reset()
            done = False
            moves = 0
            start = time.time()

            while not done:
                valid_actions = info.get("valid_actions", list(range(4)))
                action = agent.select_action(obs, valid_actions)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                moves += 1

            metrics = EpisodeMetrics(
                episode=ep,
                total_score=info.get("score", 0),
                max_tile=info.get("max_tile", 0),
                num_moves=moves,
                valid_moves=moves,
                invalid_moves=0,
                wall_clock_seconds=time.time() - start,
            )
            logger.log_episode(metrics)

    elif agent_type == "sac":
        from src.classical.sac_agent import DiscreteSACAgent
        agent = DiscreteSACAgent()
        agent.load(model_path)

        for ep in range(1, n_episodes + 1):
            obs, info = env.reset()
            done = False
            moves = 0
            start = time.time()

            while not done:
                valid_actions = info.get("valid_actions", list(range(4)))
                action = agent.select_action(obs, valid_actions, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                moves += 1

            metrics = EpisodeMetrics(
                episode=ep,
                total_score=info.get("score", 0),
                max_tile=info.get("max_tile", 0),
                num_moves=moves,
                valid_moves=moves,
                invalid_moves=0,
                wall_clock_seconds=time.time() - start,
            )
            logger.log_episode(metrics)

    elif agent_type in ("ppo", "a2c", "qrdqn"):
        if agent_type == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        elif agent_type == "a2c":
            from stable_baselines3 import A2C
            model = A2C.load(model_path)
        else:  # qrdqn
            from sb3_contrib import QRDQN
            model = QRDQN.load(model_path)


        for ep in range(1, n_episodes + 1):
            obs, info = env.reset()
            done = False
            moves = 0
            start = time.time()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                moves += 1

            metrics = EpisodeMetrics(
                episode=ep,
                total_score=info.get("score", 0),
                max_tile=info.get("max_tile", 0),
                num_moves=moves,
                valid_moves=moves,
                invalid_moves=0,
                wall_clock_seconds=time.time() - start,
            )
            logger.log_episode(metrics)

    env.close()
    logger.plot_training_curves()
    return logger.get_summary()


def main():
    parser = argparse.ArgumentParser(description="2048 Classical RL Training")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_p = sub.add_parser("train", help="Train an agent")
    train_p.add_argument(
        "--agent", choices=["dqn", "ppo", "a2c", "qrdqn", "sac"], required=True, help="Agent type"
    )
    train_p.add_argument(
        "--steps", type=int, default=500_000, help="Total training steps"
    )
    train_p.add_argument(
        "--reward", default="score_delta",
        choices=["score_delta", "log_score", "shaped"],
        help="Reward mode",
    )
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--log-dir", default="logs")
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument(
        "--n-envs", type=int, default=1,
        help="Number of parallel envs for PPO/A2C (VecEnv). Default: 1.",
    )

    # Eval command
    eval_p = sub.add_parser("eval", help="Evaluate a trained agent")
    eval_p.add_argument(
        "--agent", choices=["dqn", "ppo", "a2c", "qrdqn", "sac"], required=True, help="Agent type"
    )
    eval_p.add_argument("--model", required=True, help="Path to model file")
    eval_p.add_argument("--episodes", type=int, default=100)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--render", action="store_true")
    eval_p.add_argument("--log-dir", default="logs/eval")

    args = parser.parse_args()

    if args.command == "train":
        log_dir = os.path.join(args.log_dir, args.agent)
        if args.agent == "dqn":
            from src.classical.dqn_agent import train_dqn
            kwargs = {}
            if args.lr:
                kwargs["lr"] = args.lr
            train_dqn(
                total_steps=args.steps,
                log_dir=log_dir,
                reward_mode=args.reward,
                seed=args.seed,
                **kwargs,
            )
        elif args.agent == "ppo":
            from src.classical.ppo_agent import train_ppo
            kwargs = {}
            if args.lr:
                kwargs["lr"] = args.lr
            train_ppo(
                total_steps=args.steps,
                log_dir=log_dir,
                reward_mode=args.reward,
                seed=args.seed,
                n_envs=args.n_envs,
                **kwargs,
            )
        elif args.agent == "a2c":
            from src.classical.a2c_agent import train_a2c
            kwargs = {}
            if args.lr:
                kwargs["lr"] = args.lr
            train_a2c(
                total_steps=args.steps,
                log_dir=log_dir,
                reward_mode=args.reward,
                seed=args.seed,
                n_envs=args.n_envs,
                **kwargs,
            )
        elif args.agent == "qrdqn":
            from src.classical.qrdqn_agent import train_qrdqn
            kwargs = {}
            if args.lr:
                kwargs["lr"] = args.lr
            train_qrdqn(
                total_steps=args.steps,
                log_dir=log_dir,
                reward_mode=args.reward,
                seed=args.seed,
                **kwargs,
            )
        elif args.agent == "sac":
            from src.classical.sac_agent import train_sac
            kwargs = {}
            if args.lr:
                kwargs["lr"] = args.lr
            train_sac(
                total_steps=args.steps,
                log_dir=log_dir,
                reward_mode=args.reward,
                seed=args.seed,
                **kwargs,
            )

    elif args.command == "eval":
        summary = evaluate_agent(
            agent_type=args.agent,
            model_path=args.model,
            n_episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            log_dir=args.log_dir,
        )
        print("\nEvaluation Results:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
