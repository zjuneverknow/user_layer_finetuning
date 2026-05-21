import argparse
import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Dict

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


TARGET_MODULE_PRESETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qv": ["q_proj", "v_proj"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="data_user_state/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data_user_state/valid.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["lora", "rslora", "dora", "dora_rslora"])
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="all", choices=sorted(TARGET_MODULE_PRESETS))
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def get_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULE_PRESETS[args.target_modules],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=args.method in {"rslora", "dora_rslora"},
        use_dora=args.method in {"dora", "dora_rslora"},
    )


def filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if is_dataclass(cls):
        allowed = {field.name for field in fields(cls)}
    else:
        allowed = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    common_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_seq_length,
        "packing": False,
    }

    config_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    accepted = filter_kwargs(config_cls, common_kwargs)

    if "eval_strategy" in filter_kwargs(config_cls, {"eval_strategy": "steps"}):
        accepted["eval_strategy"] = "steps"
    elif "evaluation_strategy" in filter_kwargs(config_cls, {"evaluation_strategy": "steps"}):
        accepted["evaluation_strategy"] = "steps"

    return config_cls(**accepted)


def build_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    dataset: Any,
    peft_config: LoraConfig,
) -> SFTTrainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": getattr(training_args, "max_seq_length", None) or getattr(training_args, "max_length", None),
        "packing": False,
    }
    signature = inspect.signature(SFTTrainer.__init__).parameters
    return SFTTrainer(**{key: value for key, value in kwargs.items() if key in signature and value is not None})


def main() -> None:
    args = parse_args()

    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.valid_file})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(args),
        device_map="auto",
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=build_training_args(args),
        dataset=dataset,
        peft_config=build_peft_config(args),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
