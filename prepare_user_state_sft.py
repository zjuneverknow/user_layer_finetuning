import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


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


SYSTEM_PROMPT = (
    "你是一个对话用户状态识别模型。"
    "根据多轮对话识别当前用户层状态。"
    "只输出合法 JSON，不要解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Raw JSONL file. If omitted, raw_dataset.jsonl or dataset.jsonl is used.",
    )
    parser.add_argument("--output_dir", type=str, default="data_user_state")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_input_path(input_file: str | None) -> Path:
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in ("raw_dataset.jsonl", "dataset.jsonl"):
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError("No input file found. Expected raw_dataset.jsonl or dataset.jsonl.")


def format_dialogue(dialogue_window: List[Dict[str, str]]) -> str:
    lines = []
    for turn in dialogue_window:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def extract_user_state(labels: Dict[str, Any]) -> Dict[str, Any]:
    user_state = {}
    for key in USER_STATE_KEYS:
        if key not in labels:
            raise ValueError(f"Missing key in labels: {key}")
        user_state[key] = labels[key]

    for key in ("valence", "arousal"):
        value = float(user_state[key])
        user_state[key] = max(0.0, min(1.0, value))

    return user_state


def build_text(example: Dict[str, Any]) -> Dict[str, Any]:
    dialogue = format_dialogue(example["dialogue_window"])
    user_state = extract_user_state(example["labels"])
    output_json = json.dumps(user_state, ensure_ascii=False, separators=(",", ":"))

    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "请识别以下对话窗口中的用户状态：\n"
        f"{dialogue}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output_json}"
        "<|im_end|>"
    )

    return {
        "id": example.get("id"),
        "text": text,
        "target": output_json,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Ratios must satisfy train_ratio > 0, valid_ratio >= 0, and sum < 1.")

    input_path = resolve_input_path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_jsonl(input_path)
    processed = []
    skipped = 0

    for ex in raw_data:
        try:
            processed.append(build_text(ex))
        except Exception as exc:
            skipped += 1
            print(f"[skip] {ex.get('id')} because {exc}")

    random.seed(args.seed)
    random.shuffle(processed)

    n_total = len(processed)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)

    train_data = processed[:n_train]
    valid_data = processed[n_train : n_train + n_valid]
    test_data = processed[n_train + n_valid :]

    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(valid_data, output_dir / "valid.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")

    print(f"input_file: {input_path}")
    print(f"total raw: {len(raw_data)}")
    print(f"processed: {len(processed)}")
    print(f"skipped: {skipped}")
    print(f"train: {len(train_data)}")
    print(f"valid: {len(valid_data)}")
    print(f"test: {len(test_data)}")


if __name__ == "__main__":
    main()
