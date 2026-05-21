#!/usr/bin/env bash
set -euo pipefail

python download_model.py \
  --model_name Qwen/Qwen3.5-0.8B \
  --output_dir models/Qwen3.5-0.8B
