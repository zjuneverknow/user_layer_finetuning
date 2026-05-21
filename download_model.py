import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=args.model_name,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded model to: {local_path}")
    print("Use this local path for training:")
    print(f"  --model_name {output_dir}")


if __name__ == "__main__":
    main()
