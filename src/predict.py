from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from dataset import get_eval_transform
from model import build_model
from utils import get_device


DISCLAIMER = (
    "This is for research/demo purposes only and is not a medical diagnosis."
)
SAFE_UNCERTAIN_OUTPUT = "uncertain_tumor_review_recommended"
SAFE_UNCERTAIN_MESSAGE = "AI result: Uncertain pattern. Medical review recommended."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict brain MRI class for one image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to an image file.")
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a trained checkpoint, e.g. outputs/checkpoints/best_model.pth.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


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
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    confidence, predicted_index = torch.max(probabilities, dim=0)
    return class_names[predicted_index.item()], confidence.item(), probabilities.tolist()


def tumor_probability(class_names: list[str], probabilities: list[float]) -> float:
    return sum(
        probability
        for class_name, probability in zip(class_names, probabilities)
        if class_name != "notumor"
    )


def notumor_probability(class_names: list[str], probabilities: list[float]) -> float:
    return probabilities[class_names.index("notumor")]


def safety_gate_triggers(predicted_class: str, tumor_prob: float) -> bool:
    return (
        config.ENABLE_SAFETY_GATE
        and predicted_class == "notumor"
        and tumor_prob >= config.TUMOR_PROB_ALERT_THRESHOLD
    )


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = get_device()
    model, class_names = load_model(checkpoint_path, device)
    predicted_class, confidence, probabilities = predict_image(
        model,
        image_path,
        class_names,
        device,
    )
    tumor_prob = tumor_probability(class_names, probabilities)
    notumor_prob = notumor_probability(class_names, probabilities)
    gate_triggered = safety_gate_triggers(predicted_class, tumor_prob)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.4f}")
    print(f"Tumor probability: {tumor_prob:.4f}")
    print(f"Notumor probability: {notumor_prob:.4f}")
    print(f"Safety gate enabled: {config.ENABLE_SAFETY_GATE}")
    print(f"Tumor probability alert threshold: {config.TUMOR_PROB_ALERT_THRESHOLD:.4f}")
    if gate_triggered:
        print(f"Safe output: {SAFE_UNCERTAIN_OUTPUT}")
        print(SAFE_UNCERTAIN_MESSAGE)
    else:
        print(f"Safe output: {predicted_class}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
