"""
Gymnasium wrapper for the 2048 game engine.

Provides a standard Gymnasium-compatible environment for classical RL agents.

Observation:
    Box(0.0, 1.0, shape=(16, 4, 4), dtype=float32)
    16-channel binary tensor — channel i is active (=1) at positions
    where the tile value equals 2^(i+1).

Action space:
    Discrete(4)  →  0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

Reward modes:
    score_delta : raw merge score per step (default)
    log_score   : log2-scaled score delta
    shaped      : score_delta + monotonicity shaping

Info dict (per step):
    valid (bool)        : move changed the board
    score (int)         : total game score
    max_tile (int)      : highest tile on the board
    valid_actions (list): list of currently-valid action ints
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics

from src.env.game_2048 import Game2048


class Gym2048Env(gym.Env):
    """
    Gymnasium environment for 2048.

    Parameters
    ----------
    reward_mode : str
        One of 'score_delta', 'log_score', 'shaped'.
    seed : int or None
        RNG seed for reproducibility.
    max_steps : int or None
        Maximum steps per episode before truncation. None = unlimited.
    render_mode : str or None
        'human' for text rendering to stdout.
    invalid_move_penalty : float
        Reward applied on invalid moves (default −0.5).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    REWARD_MODES = ("score_delta", "log_score", "shaped")

    def __init__(
        self,
        reward_mode: str = "score_delta",
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        invalid_move_penalty: float = -0.5,
    ):
        super().__init__()
        if reward_mode not in self.REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {self.REWARD_MODES}")

        self.reward_mode = reward_mode
        self._seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.invalid_move_penalty = invalid_move_penalty

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16, 4, 4), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Internal game engine
        self.game = Game2048(seed=seed)
        self._steps = 0

    # ─── Gymnasium API ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self.game.reset(seed=effective_seed)
        self._steps = 0
        obs = self.game.to_obs()
        info = self._make_info(valid=True)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action to the environment.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        self._steps += 1

        _, score_delta, terminated, step_info = self.game.step(int(action))
        valid = step_info["valid"]

        # Reward computation
        if not valid:
            reward = self.invalid_move_penalty
        else:
            reward = self._compute_reward(score_delta)

        # Truncation
        truncated = (
            self.max_steps is not None and self._steps >= self.max_steps
        ) and not terminated

        obs = self.game.to_obs()
        info = self._make_info(valid=valid, score_delta=score_delta)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        rendered = self.game.render()
        if self.render_mode == "human":
            print(rendered)
        return rendered

    def close(self) -> None:
        pass

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _compute_reward(self, score_delta: float) -> float:
        if self.reward_mode == "score_delta":
            return float(score_delta)

        if self.reward_mode == "log_score":
            if score_delta <= 0:
                return 0.0
            return float(np.log2(score_delta + 1))

        if self.reward_mode == "shaped":
            base = float(score_delta)
            mono = self._monotonicity_bonus()
            return base + mono

        return float(score_delta)

    def _monotonicity_bonus(self) -> float:
        """
        Small bonus encouraging monotonic rows/columns.

        Measures how many adjacent pairs within each row/column are in
        non-decreasing order; scales 0→0.5.
        """
        board = self.game.board.astype(np.float32)
        # Replace zeros with 1 for log-stability (log2(1)=0)
        log_board = np.where(board > 0, np.log2(np.maximum(board, 1)), 0.0)

        score = 0.0
        # Rows
        for row in log_board:
            score += np.sum(np.diff(row) >= 0)
        # Columns
        for col in log_board.T:
            score += np.sum(np.diff(col) >= 0)

        max_pairs = 2 * (log_board.shape[0] - 1) * log_board.shape[0]  # 24
        return 0.5 * score / max_pairs

    def _make_info(
        self, valid: bool = True, score_delta: float = 0.0
    ) -> dict:
        return {
            "valid": valid,
            "score": self.game.score,
            "max_tile": self.game.max_tile,
            "valid_actions": self.game.get_valid_actions(),
            "score_delta": score_delta,
        }
