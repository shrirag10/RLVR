# Solving 2048: Classical RL vs. LLM-Guided RLVR

> **CS 5180 — Reinforcement Learning** | Shriman + Partner

A comparative study of classical Reinforcement Learning (DQN, QR-DQN, PPO, A2C, SAC) versus LLM-based Reinforcement Learning with Verifiable Rewards (GRPO with Qwen2.5-0.5B) on the puzzle game 2048.

## Project Structure

```
RLVR/
├── src/
│   ├── env/
│   │   ├── game_2048.py        # Core 2048 engine (pure Python)
│   │   ├── gym_wrapper.py      # Gymnasium env — 16-channel CNN input, 3 reward modes
│   │   └── text_wrapper.py     # Text/prompt wrapper for LLM + response parser
│   ├── classical/
│   │   ├── dqn_agent.py        # Custom DQN (CNN, replay buffer, target net)
│   │   ├── ppo_agent.py        # SB3 PPO with CnnPolicy
│   │   ├── a2c_agent.py        # SB3 A2C
│   │   ├── qrdqn_agent.py      # SB3-contrib QR-DQN
│   │   ├── sac_agent.py        # Discrete SAC
│   │   └── train.py            # Unified train/eval CLI for all classical agents
│   ├── llm/
│   │   ├── reward.py           # Multi-component verifiable reward functions
│   │   ├── dataset.py          # Board state dataset generator (HuggingFace Dataset)
│   │   ├── train_grpo.py       # GRPO training pipeline (Unsloth + TRL)
│   │   ├── prompt.py           # Prompt templates
│   │   └── predict.py          # LLM inference utilities
│   └── utils/
│       └── metrics.py          # EpisodeMetrics, TrainingLogger, matplotlib plots
├── configs/
│   ├── dqn_config.yaml
│   ├── ppo_config.yaml
│   └── grpo_config.yaml        # Full GRPO hyperparameter config
├── tests/
│   ├── test_env.py
│   └── test_reward.py
├── report/                     # LaTeX final report + training curve figures
└── requirements/
    ├── base.txt
    ├── classical.txt
    └── llm.txt
```

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements/base.txt

# Run tests
python -m pytest tests/ -v

# Train DQN (Track A)
pip install -r requirements/classical.txt
python -m src.classical.train train --agent dqn --steps 500000

# Train PPO (Track A)
python -m src.classical.train train --agent ppo --steps 1000000

# Train with other classical agents
python -m src.classical.train train --agent a2c --steps 500000
python -m src.classical.train train --agent qrdqn --steps 500000
python -m src.classical.train train --agent sac --steps 500000

# Evaluate a trained agent
python -m src.classical.train eval --agent dqn --model logs/dqn/dqn_final.pt --episodes 100

# Train LLM (Track B) — requires GPU
pip install -r requirements/llm.txt
python -m src.llm.train_grpo --dataset-size 10000 --epochs 3

# Train LLM from config file
python -m src.llm.train_grpo --config configs/grpo_config.yaml
```

## Tracks

| Track | Agent | Method | Framework |
|-------|-------|--------|-----------|
| A | DQN | Custom CNN + Replay Buffer + Target Net | PyTorch |
| A | QR-DQN | Quantile Regression DQN | SB3-contrib |
| A | PPO | CnnPolicy + Custom Feature Extractor | Stable-Baselines3 |
| A | A2C | CnnPolicy | Stable-Baselines3 |
| A | SAC | Discrete SAC | PyTorch |
| B | LLM | Qwen2.5-0.5B + GRPO + QLoRA 4-bit | Unsloth + TRL |

## Environment

The 2048 game environment has two interfaces:

- **Gym wrapper** (`gym_wrapper.py`): Standard Gymnasium env for classical RL. Observations are 16-channel binary tensors (one channel per tile value, 2–32768). Supports three reward modes:
  - `score_delta` — raw merge score per step
  - `log_score` — log2-scaled score delta
  - `shaped` — additional shaping signal

- **Text wrapper** (`text_wrapper.py`): Converts board states to structured text prompts for the LLM. Parses model outputs in the format:
  ```
  <think>
  [reasoning about the board]
  </think>
  <answer>UP|DOWN|LEFT|RIGHT</answer>
  ```

## LLM Track — GRPO Training

The LLM track fine-tunes **Qwen2.5-0.5B-Instruct** using GRPO (Group Relative Policy Optimization) with QLoRA 4-bit via Unsloth. The model is trained to reason about 2048 board states and select moves via chain-of-thought, following the DeepSeek-R1/TinyZero approach of prefilling the assistant turn with `<think>`.

### Dataset

10,000 board states generated via random self-play, stratified across game stages:

| Stage | Max Tile | Fraction |
|-------|----------|----------|
| Early | < 64 | 30% |
| Mid | 64–512 | 40% |
| Late | > 512 | 30% |

### Reward Functions

Five reward functions are passed separately to `GRPOTrainer` with the following weights:

| Function | Weight | Signal |
|----------|--------|--------|
| `format_reward_fn` | 0.3 | +0.5 for valid `<think>` + `<answer>` XML; +0.25 partial |
| `direction_reward_fn` | 0.3 | +0.5 for valid direction ∈ {UP, DOWN, LEFT, RIGHT} |
| `game_reward_fn` | 0.2 | +1.0 valid move, +score_delta/1024, −2.0 no-op, milestone bonuses |
| `thinking_quality_reward_fn` | 0.1 | Strategic vocabulary, tile mentions, ideal 20–150 word length |
| `length_reward_fn` | 0.1 | Soft anti-degenerate-completion signal (scales 0→0.3 over 10–50 words) |

**Milestone bonuses** (awarded once per newly reached tile):

| Tile | Bonus |
|------|-------|
| 256 | +2.0 |
| 512 | +3.0 |
| 1024 | +5.0 |
| 2048 | +10.0 |
| 4096 | +15.0 |

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| LoRA r / α | 16 / 16 |
| Group size G | 8 |
| Max completion | 256 tokens |
| Max sequence | 512 tokens |
| Learning rate | 5e-6 (cosine, 10% warmup) |
| Optimizer | AdamW 8-bit |
| Batch size | 1 (× 4 grad accum) |
| Epochs | 3 |
| Temperature | 0.9 |

## Metrics

- Max tile reached (distribution)
- Average episode score
- Sample efficiency (score/steps)
- Win rate (% reaching 2048)
- Reach rate for 512 and 1024 tiles
- Wall-clock training time
- Reasoning quality (LLM track)

Logs are saved as CSV per run; training curves (score, max tile distribution, max tile over time, moves per episode) are auto-generated as `.png` files via `TrainingLogger`.

## Hardware

- **GPU**: NVIDIA RTX 4060 (6GB VRAM)
- **GRPO VRAM**: ~3–4GB (0.5B model, QLoRA 4-bit, G=8)

## References

1. Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning via RL.
2. Wen et al. (2025). RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs.
3. Saligram et al. (2025). 2048: RL in a Delayed Reward Environment (arXiv:2507.05465).
4. Hu et al. (2025). lmgame-Bench: How Good are LLMs at Playing Games?
