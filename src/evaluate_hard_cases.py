from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

import config
from dataset import get_eval_transform
from model import build_model
from utils import get_device


HARD_CASE_CSV = config.REPORT_DIR / "combined_gate_missed_high_risk.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an experiment on hard-case audit samples only.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint if args.checkpoint.is_absolute() else config.PROJECT_ROOT / args.checkpoint
    return config.CHECKPOINT_DIR / f"best_model_{args.experiment_name}.pth"


def apply_checkpoint_config(checkpoint_config: dict[str, object], experiment_name: str) -> str:
    model_name = str(checkpoint_config.get("model_name", config.MODEL_NAME))
    config.MODEL_NAME = model_name
    config.IMAGE_SIZE = int(checkpoint_config.get("image_size", config.IMAGE_SIZE))
    config.USE_PAD_TO_SQUARE = bool(checkpoint_config.get("use_pad_to_square", config.USE_PAD_TO_SQUARE))
    config.USE_HORIZONTAL_FLIP = bool(
        checkpoint_config.get("use_horizontal_flip", config.USE_HORIZONTAL_FLIP)
    )
    config.USE_CLAHE = bool(checkpoint_config.get("use_clahe", config.USE_CLAHE))
    config.EXPERIMENT_NAME = experiment_name
    return model_name


def read_hard_cases() -> list[dict[str, str]]:
    if not HARD_CASE_CSV.exists():
        raise FileNotFoundError(
            f"Hard-case CSV not found: {HARD_CASE_CSV}. "
            "Run `python src\\export_combined_gate_misses.py` first."
        )

    with HARD_CASE_CSV.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, list[float]]:
    transform = get_eval_transform()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, predicted_idx = probabilities.max(dim=0)
    return class_names[predicted_idx.item()], confidence.item(), probabilities.tolist()


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Experiment checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    model_name = apply_checkpoint_config(checkpoint_config, args.experiment_name)

    class_names = checkpoint.get("class_names", config.CLASS_NAMES)
    device = get_device()
    model = build_model(num_classes=len(class_names), model_name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = read_hard_cases()
    still_notumor = 0
    recovered = 0
    per_sample_lines: list[str] = []

    for row in rows:
        image_path = config.PROJECT_ROOT / row["image_path"]
        prediction, confidence, probabilities = predict_image(
            model,
            image_path,
            class_names,
            device,
        )
        if prediction == "notumor":
            still_notumor += 1
        else:
            recovered += 1

        probability_text = ", ".join(
            f"{class_name}={probability:.4f}"
            for class_name, probability in zip(class_names, probabilities)
        )
        per_sample_lines.append(
            f"{row['image_path']} | true={row['true_class']} | "
            f"pred={prediction} | confidence={confidence:.4f} | {probability_text}"
        )

    report_path = config.REPORT_DIR / f"hard_case_report_{args.experiment_name}.txt"
    report_lines = [
        "Hard-Case Evaluation Report",
        f"Experiment: {args.experiment_name}",
        f"Checkpoint: {checkpoint_path}",
        f"Model: {model_name}",
        f"Image size: {config.IMAGE_SIZE}",
        f"Pad to square: {config.USE_PAD_TO_SQUARE}",
        f"Horizontal flip in training config: {config.USE_HORIZONTAL_FLIP}",
        f"CLAHE: {config.USE_CLAHE}",
        "",
        f"Hard-case samples evaluated: {len(rows)}",
        f"Still predicted as notumor: {still_notumor}",
        f"Recovered as tumor class: {recovered}",
        "",
        "Per-sample predictions:",
        *per_sample_lines,
        "",
        "Safety note:",
        "  These hard cases remain an audit set only and must not be used for training.",
        "  This report is not clinical validation.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Hard-case report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
