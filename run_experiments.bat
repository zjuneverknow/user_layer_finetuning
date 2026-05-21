@echo off
setlocal

python prepare_user_state_sft.py
if errorlevel 1 exit /b %errorlevel%

python train_sft.py --method lora --output_dir outputs/lora_r16 --r 16 --alpha 32 --dropout 0.05 --num_train_epochs 3 --learning_rate 1e-4 --max_seq_length 1024 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --bf16
if errorlevel 1 exit /b %errorlevel%

python train_sft.py --method rslora --output_dir outputs/rslora_r16 --r 16 --alpha 32 --dropout 0.05 --num_train_epochs 3 --learning_rate 1e-4 --max_seq_length 1024 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --bf16
if errorlevel 1 exit /b %errorlevel%

python train_sft.py --method dora --output_dir outputs/dora_r16 --r 16 --alpha 32 --dropout 0.05 --num_train_epochs 3 --learning_rate 1e-4 --max_seq_length 1024 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --bf16
if errorlevel 1 exit /b %errorlevel%

python train_sft.py --method dora_rslora --output_dir outputs/dora_rslora_r16 --r 16 --alpha 32 --dropout 0.05 --num_train_epochs 3 --learning_rate 1e-4 --max_seq_length 1024 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --bf16
if errorlevel 1 exit /b %errorlevel%

python evaluate_user_state.py --adapter_dir outputs/lora_r16 --output_file results/lora_r16_eval.json --bf16
if errorlevel 1 exit /b %errorlevel%

python evaluate_user_state.py --adapter_dir outputs/rslora_r16 --output_file results/rslora_r16_eval.json --bf16
if errorlevel 1 exit /b %errorlevel%

python evaluate_user_state.py --adapter_dir outputs/dora_r16 --output_file results/dora_r16_eval.json --bf16
if errorlevel 1 exit /b %errorlevel%

python evaluate_user_state.py --adapter_dir outputs/dora_rslora_r16 --output_file results/dora_rslora_r16_eval.json --bf16
if errorlevel 1 exit /b %errorlevel%

python collect_metrics.py
if errorlevel 1 exit /b %errorlevel%

echo Done. Metrics saved to results\metrics.csv
