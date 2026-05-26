from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from binary_dataset import get_binary_eval_transform
from dataset import get_eval_transform
from model import build_model
from utils import get_device


FOUR_CLASS_CHECKPOINT = config.CHECKPOINT_DIR / "best_model_manifest.pth"
BINARY_CHECKPOINT = config.CHECKPOINT_DIR / "best_binary_tumor_gate.pth"
DISCLAIMER = "This is for research/demo purposes only and is not a medical diagnosis."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with combined binary tumor gate.")
    parser.add_argument("--image", required=True, type=Path)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def load_model(
    checkpoint_path: Path,
    default_class_names: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names", default_class_names)
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict(
    model: torch.nn.Module,
    transform,
    image_path: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, list[float]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, predicted_idx = probabilities.max(dim=0)
    return class_names[predicted_idx.item()], confidence.item(), probabilities.tolist()


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    four_model, four_classes = load_model(FOUR_CLASS_CHECKPOINT, config.CLASS_NAMES, device)
    binary_model, binary_classes = load_model(
        BINARY_CHECKPOINT,
        config.BINARY_CLASS_NAMES,
        device,
    )

    four_pred, four_conf, four_probs = predict(
        four_model,
        get_eval_transform(),
        image_path,
        four_classes,
        device,
    )
    binary_pred, binary_conf, binary_probs = predict(
        binary_model,
        get_binary_eval_transform(),
        image_path,
        binary_classes,
        device,
    )

    gate_triggers = four_pred == "notumor" and binary_pred == "tumor"
    safe_output = "uncertain_tumor_review_recommended" if gate_triggers else four_pred

    print(f"4-class prediction: {four_pred}")
    print(f"4-class confidence: {four_conf:.4f}")
    for class_name, probability in zip(four_classes, four_probs):
        print(f"  probability_{class_name}: {probability:.4f}")
    print(f"Binary gate prediction: {binary_pred}")
    print(f"Binary gate confidence: {binary_conf:.4f}")
    for class_name, probability in zip(binary_classes, binary_probs):
        print(f"  binary_probability_{class_name}: {probability:.4f}")
    print(f"Combined safe output: {safe_output}")
    if gate_triggers:
        print("AI result: Uncertain pattern. Medical review recommended.")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
