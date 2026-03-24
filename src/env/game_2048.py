"""
Core 2048 game engine — pure Python/NumPy, no dependencies on RL frameworks.

This module implements the complete 2048 game logic:
- Sliding and merging tiles in four directions
- Spawning new tiles after each valid move
- Detecting game-over states
- Serialization helpers for both gym and text wrappers

Action space:
    0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class Game2048:
    """
    Core 2048 game engine.

    Attributes:
        board: 4×4 numpy array of tile values (int32). 0 = empty.
        score: Cumulative score.
        done: True when no valid moves remain.
        rng: numpy random generator (seeded).
    """

    # ─── Action Constants ─────────────────────────────────────────────
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    NAME_TO_ACTION = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

    # Tile values for the 16-channel binary observation:
    #   Channel 0  = empty cell (value 0)
    #   Channel i (1..15) = tile value 2^i  (2, 4, 8, ..., 32768)
    # Every cell is always active in exactly one channel.
    TILE_VALUES = [2 ** i for i in range(1, 16)]  # 2, 4, 8, ..., 32768  (15 values)

    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()

    # ─── Public API ───────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> "Game2048":
        """Reset the board to initial state and return self."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute a move.

        Args:
            action: One of UP=0, RIGHT=1, DOWN=2, LEFT=3.

        Returns:
            (board, score_delta, done, info)
            info keys: valid (bool), merges (int), max_tile (int),
                        total_score (int)
        """
        old_board = self.board.copy()
        old_score = self.score

        new_board, score_delta, merges = self._slide(self.board.copy(), action)

        valid = not np.array_equal(old_board, new_board)

        if valid:
            self.board = new_board
            self.score += score_delta
            self._spawn_tile()

        if not self._has_valid_moves():
            self.done = True

        info = {
            "valid": valid,
            "merges": merges,
            "max_tile": int(np.max(self.board)),
            "total_score": self.score,
            "score": self.score,
        }
        return self.board.copy(), float(score_delta if valid else 0.0), self.done, info

    def is_valid_action(self, action: int) -> bool:
        """Return True if the action would change the board."""
        new_board, _, _ = self._slide(self.board.copy(), action)
        return not np.array_equal(self.board, new_board)

    def get_valid_actions(self) -> list[int]:
        """Return list of actions that would change the board."""
        return [a for a in range(4) if self.is_valid_action(a)]

    def _has_valid_moves(self) -> bool:
        """Return True if at least one action is valid."""
        return any(self.is_valid_action(a) for a in range(4))

    def clone(self) -> "Game2048":
        """Return an independent deep copy of this game."""
        g = Game2048.__new__(Game2048)
        g.size = self.size
        g.board = self.board.copy()
        g.score = self.score
        g.done = self.done
        g.rng = copy.deepcopy(self.rng)
        return g

    def render(self) -> str:
        """Return a pretty-printed string representation."""
        lines = [f"Score: {self.score}  Max: {int(np.max(self.board))}"]
        lines.append("+" + "------+" * self.size)
        for row in self.board:
            cells = "".join(f"{v:^6}|" if v > 0 else "      |" for v in row)
            lines.append("|" + cells)
            lines.append("+" + "------+" * self.size)
        return "\n".join(lines)

    def to_list(self) -> list[list[int]]:
        """Return board as nested Python list (row-major)."""
        return self.board.tolist()

    def to_obs(self) -> np.ndarray:
        """
        Encode board as 16-channel binary tensor for CNN input.

        Shape: (16, 4, 4), dtype float32.
        Channel 0  = 1 where cell is empty (value 0).
        Channel i (1..15) = 1 where board value == 2^i.

        Every cell is active in exactly one channel, so obs.sum(axis=0)
        is all-ones — the invariant checked by the test suite.
        """
        obs = np.zeros((16, self.size, self.size), dtype=np.float32)
        # Channel 0: empty cells
        obs[0] = (self.board == 0).astype(np.float32)
        # Channels 1..15: tile values 2^1 .. 2^15
        for i, tile_val in enumerate(self.TILE_VALUES, start=1):
            obs[i] = (self.board == tile_val).astype(np.float32)
        return obs

    @property
    def max_tile(self) -> int:
        return int(np.max(self.board))

    # ─── Internal Mechanics ───────────────────────────────────────────

    def _spawn_tile(self) -> None:
        """Spawn a 2 (90%) or 4 (10%) on a random empty cell."""
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        row, col = empty[int(self.rng.integers(0, len(empty)))]
        self.board[row, col] = 4 if self.rng.random() < 0.1 else 2

    def _slide_row_left(self, row: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Slide a single row to the left.

        Returns:
            (new_row, score_delta, merges)
        """
        # Compact non-zero tiles to the left
        tiles = row[row != 0]
        new_row = np.zeros(self.size, dtype=np.int32)
        score_delta = 0
        merges = 0
        write = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                # Merge
                merged = tiles[i] * 2
                new_row[write] = merged
                score_delta += merged
                merges += 1
                write += 1
                i += 2
            else:
                new_row[write] = tiles[i]
                write += 1
                i += 1
        return new_row, score_delta, merges

    def _slide(
        self, board: np.ndarray, action: int
    ) -> tuple[np.ndarray, int, int]:
        """
        Apply a slide action to the board.

        Strategy: rotate the board so the target direction is always LEFT,
        apply a left-slide to each row, then rotate back.

        Rotations:
            UP    → rotate 90° CW  → slide left → rotate 90° CCW
            RIGHT → rotate 180°    → slide left → rotate 180°
            DOWN  → rotate 90° CCW → slide left → rotate 90° CW
            LEFT  → no rotation
        """
        # Number of 90° CCW rotations to bring target direction to LEFT
        rot_map = {self.UP: 1, self.RIGHT: 2, self.DOWN: 3, self.LEFT: 0}
        k = rot_map[action]

        # np.rot90 rotates CCW by default
        board = np.rot90(board, k)

        total_score = 0
        total_merges = 0
        for r in range(self.size):
            new_row, sc, mg = self._slide_row_left(board[r])
            board[r] = new_row
            total_score += sc
            total_merges += mg

        # Rotate back: k CCW → (4-k) CCW
        board = np.rot90(board, (4 - k) % 4)
        return board, total_score, total_merges
