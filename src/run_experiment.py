from __future__ import annotations

import argparse
import importlib
import os

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one 4-class experiment.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument("--pad-to-square", action="store_true")
    parser.add_argument("--no-horizontal-flip", action="store_true")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--experiment-name", required=True)
    return parser.parse_args()


def set_runtime_config(args: argparse.Namespace) -> None:
    values = {
        "MODEL_NAME": args.model_name,
        "IMAGE_SIZE": str(args.image_size),
        "USE_PAD_TO_SQUARE": str(args.pad_to_square),
        "USE_HORIZONTAL_FLIP": str(not args.no_horizontal_flip),
        "USE_CLAHE": str(args.use_clahe),
        "EXPERIMENT_NAME": args.experiment_name,
    }
    os.environ.update(values)

    config.MODEL_NAME = args.model_name
    config.IMAGE_SIZE = args.image_size
    config.USE_PAD_TO_SQUARE = args.pad_to_square
    config.USE_HORIZONTAL_FLIP = not args.no_horizontal_flip
    config.USE_CLAHE = args.use_clahe
    config.EXPERIMENT_NAME = args.experiment_name


def main() -> int:
    args = parse_args()
    set_runtime_config(args)

    print("Running experiment with config:")
    print(f"  EXPERIMENT_NAME={config.EXPERIMENT_NAME}")
    print(f"  MODEL_NAME={config.MODEL_NAME}")
    print(f"  IMAGE_SIZE={config.IMAGE_SIZE}")
    print(f"  USE_PAD_TO_SQUARE={config.USE_PAD_TO_SQUARE}")
    print(f"  USE_HORIZONTAL_FLIP={config.USE_HORIZONTAL_FLIP}")
    print(f"  USE_CLAHE={config.USE_CLAHE}")

    train = importlib.import_module("train")
    train_result = train.main()
    if train_result != 0:
        return train_result

    evaluate = importlib.import_module("evaluate")
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main())
