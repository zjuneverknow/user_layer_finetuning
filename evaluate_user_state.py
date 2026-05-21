import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


USER_STATE_KEYS = [
    "dominant_emotion",
    "valence",
    "arousal",
    "activity",
    "user_behavior",
    "behavior_intensity",
    "stress_level",
    "mental_fatigue",
    "attention_need",
    "distraction_tolerance",
]

DISCRETE_KEYS = [
    "dominant_emotion",
    "activity",
    "user_behavior",
    "behavior_intensity",
    "stress_level",
    "mental_fatigue",
    "attention_need",
    "distraction_tolerance",
]

CONTINUOUS_KEYS = ["valence", "arousal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="data_user_state/test.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def get_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def extract_assistant_target(text: str) -> Dict[str, Any]:
    marker = "<|im_start|>assistant\n"
    if marker not in text:
        raise ValueError("No assistant marker")
    target_part = text.split(marker, 1)[1].replace("<|im_end|>", "").strip()
    return json.loads(target_part)


def build_prompt(text: str) -> str:
    marker = "<|im_start|>assistant\n"
    if marker not in text:
        raise ValueError("No assistant marker")
    return text.split(marker, 1)[0] + marker


def extract_json_from_output(output: str) -> Optional[Dict[str, Any]]:
    output = output.replace("<|im_end|>", "").strip()
    try:
        return json.loads(output)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", output, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def evaluate_prediction(pred: Optional[Dict[str, Any]], gold: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"json_valid": int(pred is not None)}

    if pred is None:
        result["field_complete"] = 0
        for key in DISCRETE_KEYS:
            result[f"{key}_correct"] = 0
        for key in CONTINUOUS_KEYS:
            result[f"{key}_abs_error"] = None
        result["exact_match_discrete"] = 0
        return result

    result["field_complete"] = int(all(key in pred for key in USER_STATE_KEYS))
    discrete_all_correct = True

    for key in DISCRETE_KEYS:
        correct = int(normalize_value(pred.get(key)) == normalize_value(gold.get(key)))
        result[f"{key}_correct"] = correct
        discrete_all_correct = discrete_all_correct and bool(correct)

    for key in CONTINUOUS_KEYS:
        try:
            pred_value = max(0.0, min(1.0, float(pred.get(key))))
            gold_value = max(0.0, min(1.0, float(gold.get(key))))
            result[f"{key}_abs_error"] = abs(pred_value - gold_value)
        except Exception:
            result[f"{key}_abs_error"] = None

    result["exact_match_discrete"] = int(discrete_all_correct)
    return result


def mean_ignore_none(values: List[Any]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files={"test": args.test_file})["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(args),
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    device = next(model.parameters()).device

    all_results = []
    raw_outputs = []

    for item in tqdm(dataset):
        text = item["text"]
        gold = extract_assistant_target(text)
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        pred = extract_json_from_output(generated_text)
        eval_result = evaluate_prediction(pred, gold)
        all_results.append(eval_result)
        raw_outputs.append(
            {
                "id": item.get("id"),
                "gold": gold,
                "generated_text": generated_text.replace("<|im_end|>", "").strip(),
                "parsed_pred": pred,
                "eval": eval_result,
            }
        )

    metrics: Dict[str, Any] = {
        "json_valid_rate": sum(r["json_valid"] for r in all_results) / len(all_results),
        "field_complete_rate": sum(r["field_complete"] for r in all_results) / len(all_results),
        "exact_match_discrete": sum(r["exact_match_discrete"] for r in all_results) / len(all_results),
    }

    for key in DISCRETE_KEYS:
        metrics[f"{key}_accuracy"] = sum(r[f"{key}_correct"] for r in all_results) / len(all_results)

    for key in CONTINUOUS_KEYS:
        metrics[f"{key}_mae"] = mean_ignore_none([r[f"{key}_abs_error"] for r in all_results])

    output = {
        "adapter_dir": args.adapter_dir,
        "metrics": metrics,
        "raw_outputs": raw_outputs,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
