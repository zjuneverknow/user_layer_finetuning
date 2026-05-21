#!/usr/bin/env bash
set -euo pipefail

python download_model.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir models/Qwen2.5-1.5B-Instruct
