from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import config
from binary_dataset import create_binary_dataloaders
from model import build_model
from utils import ensure_dirs, get_device


CHECKPOINT_PATH = config.CHECKPOINT_DIR / "best_binary_tumor_gate.pth"
REPORT_PATH = config.REPORT_DIR / "binary_test_report.txt"
CONFUSION_MATRIX_PATH = config.OUTPUT_DIR / "binary_confusion_matrix.png"


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Binary checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint.get("class_names", config.BINARY_CLASS_NAMES)
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    labels_all: list[int] = []
    predictions_all: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="binary-test"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            predictions_all.extend(predictions)
            labels_all.extend(labels.tolist())

    return labels_all, predictions_all


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Binary Tumor Gate Confusion Matrix",
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader, dataset_class_names = create_binary_dataloaders()
    model, class_names = load_model(device)
    if class_names != dataset_class_names:
        raise ValueError(f"Checkpoint classes {class_names} != dataset classes {dataset_class_names}")

    labels, predictions = predict_loader(model, test_loader, device)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=class_names,
        digits=4,
    )

    tumor_precision = matrix[0, 0] / max(matrix[:, 0].sum(), 1)
    tumor_recall = matrix[0, 0] / max(matrix[0, :].sum(), 1)
    notumor_precision = matrix[1, 1] / max(matrix[:, 1].sum(), 1)
    notumor_recall = matrix[1, 1] / max(matrix[1, :].sum(), 1)
    tumor_false_negatives = int(matrix[0, 1])
    notumor_false_positives = int(matrix[1, 0])

    report_text = "\n".join(
        [
            "Binary Tumor Gate Test Report",
            f"Checkpoint: {CHECKPOINT_PATH}",
            f"Binary accuracy: {accuracy:.4f}",
            f"Tumor precision: {tumor_precision:.4f}",
            f"Tumor recall: {tumor_recall:.4f}",
            f"Notumor precision: {notumor_precision:.4f}",
            f"Notumor recall: {notumor_recall:.4f}",
            f"Tumor false negatives: {tumor_false_negatives}",
            f"Notumor false positives: {notumor_false_positives}",
            "",
            report,
            "Confusion matrix [[tumor, notumor] rows x [tumor, notumor] cols]:",
            str(matrix),
            "",
            "Safety note: tumor recall and tumor false negatives are the primary metrics.",
            "This binary gate is not clinically validated.",
        ]
    )

    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    plot_confusion_matrix(matrix, class_names)
    print(report_text)
    print(f"Binary report saved to: {REPORT_PATH}")
    print(f"Binary confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
