#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen3.5-0.8B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"

mkdir -p outputs results

python train_sft.py \
  --model_name "$MODEL_PATH" \
  --method lora \
  --output_dir outputs/qwen35_08b_lora_r16 \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --max_seq_length 1024 \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --bf16 \
  --resume_from_checkpoint auto

python train_sft.py \
  --model_name "$MODEL_PATH" \
  --method rslora \
  --output_dir outputs/qwen35_08b_rslora_r16 \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --max_seq_length 1024 \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --bf16 \
  --resume_from_checkpoint auto

python train_sft.py \
  --model_name "$MODEL_PATH" \
  --method dora \
  --output_dir outputs/qwen35_08b_dora_r16 \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --max_seq_length 1024 \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --bf16 \
  --resume_from_checkpoint auto

python train_sft.py \
  --model_name "$MODEL_PATH" \
  --method dora_rslora \
  --output_dir outputs/qwen35_08b_dora_rslora_r16 \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --max_seq_length 1024 \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --bf16 \
  --resume_from_checkpoint auto

python evaluate_user_state.py \
  --base_model "$MODEL_PATH" \
  --adapter_dir outputs/qwen35_08b_lora_r16 \
  --output_file results/qwen35_08b_lora_r16_eval.json \
  --bf16

python evaluate_user_state.py \
  --base_model "$MODEL_PATH" \
  --adapter_dir outputs/qwen35_08b_rslora_r16 \
  --output_file results/qwen35_08b_rslora_r16_eval.json \
  --bf16

python evaluate_user_state.py \
  --base_model "$MODEL_PATH" \
  --adapter_dir outputs/qwen35_08b_dora_r16 \
  --output_file results/qwen35_08b_dora_r16_eval.json \
  --bf16

python evaluate_user_state.py \
  --base_model "$MODEL_PATH" \
  --adapter_dir outputs/qwen35_08b_dora_rslora_r16 \
  --output_file results/qwen35_08b_dora_rslora_r16_eval.json \
  --bf16

python collect_metrics.py \
  --preset qwen35_08b \
  --output_file results/qwen35_08b_metrics.csv

echo "Done. Metrics saved to results/qwen35_08b_metrics.csv"
