import argparse
import csv
import json
from pathlib import Path
from typing import Dict


DEFAULT_FILES = {
    "lora_r16": "results/lora_r16_eval.json",
    "rslora_r16": "results/rslora_r16_eval.json",
    "dora_r16": "results/dora_r16_eval.json",
    "dora_rslora_r16": "results/dora_rslora_r16_eval.json",
}

QWEN35_08B_FILES = {
    "qwen35_08b_lora_r16": "results/qwen35_08b_lora_r16_eval.json",
    "qwen35_08b_rslora_r16": "results/qwen35_08b_rslora_r16_eval.json",
    "qwen35_08b_dora_r16": "results/qwen35_08b_dora_r16_eval.json",
    "qwen35_08b_dora_rslora_r16": "results/qwen35_08b_dora_rslora_r16_eval.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="results/metrics.csv")
    parser.add_argument("--preset", type=str, default="default", choices=["default", "qwen35_08b"])
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def load_metrics(files: Dict[str, str], allow_missing: bool) -> list[dict]:
    rows = []
    for method, file_name in files.items():
        path = Path(file_name)
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing evaluation file: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row = {"method": method}
        row.update(data["metrics"])
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    files = QWEN35_08B_FILES if args.preset == "qwen35_08b" else DEFAULT_FILES
    rows = load_metrics(files, args.allow_missing)
    if not rows:
        raise ValueError("No metric files found.")

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = ["method"] + sorted(key for key in all_keys if key != "method")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
