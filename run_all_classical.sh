#!/usr/bin/env bash
# run_all_classical.sh — Sequential Track A training for all 5 agents
# Logs: logs/<agent>/<agent>_training.log
set -e

VENV_DIR="$(dirname "$0")/venv"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$(dirname "$0")"

LOG_ROOT="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_agent() {
    local agent=$1; shift
    local extra_args="$@"
    local log_dir="$LOG_ROOT/$agent"
    mkdir -p "$log_dir"
    local logfile="$log_dir/${agent}_${TIMESTAMP}.log"
    echo ""
    echo "=========================================="
    echo "  Starting $agent  $(date)"
    echo "  Log: $logfile"
    echo "=========================================="
    python -m src.classical.train train --agent "$agent" $extra_args 2>&1 | tee "$logfile"
    echo "  $agent done  $(date)"
}

run_agent dqn   --steps 500000  --reward score_delta
run_agent ppo   --steps 1000000 --reward score_delta --n-envs 4
run_agent a2c   --steps 500000  --reward score_delta --n-envs 8
run_agent qrdqn --steps 500000  --reward score_delta
run_agent sac   --steps 500000  --reward score_delta

echo ""
echo "=========================================="
echo "  All classical agents done!  $(date)"
echo "=========================================="
