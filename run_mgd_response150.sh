#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project="runs/mgd_response150"
name="mgd_response150_seed0"
run="$project/$name"
state="$run/weights/moe_last.pt"
mkdir -p "$project"
if [[ -f "$project/PAUSE_QUEUE" ]]; then
  printf 'Remove %s/PAUSE_QUEUE to start or resume.\n' "$project" >&2
  exit 1
fi
args=(--project "$project" --name "$name" --device 0)
if [[ -f "$state" ]]; then
  args+=(--resume-state "$state")
elif [[ -e "$run" ]]; then
  printf 'Run exists without a full-state checkpoint: %s\n' "$run" >&2
  exit 1
fi
python -u train_mgd_response150.py "${args[@]}"
