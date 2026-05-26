from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import build_model
from utils import ensure_dirs, get_device


BEST_MODEL_PATH = config.CHECKPOINT_DIR / f"best_model_{config.EXPERIMENT_NAME}.pth"
REPORT_PATH = config.REPORT_DIR / f"test_report_{config.EXPERIMENT_NAME}.txt"
CONFUSION_MATRIX_PATH = config.OUTPUT_DIR / f"confusion_matrix_{config.EXPERIMENT_NAME}.png"
MISCLASSIFIED_CSV_PATH = config.REPORT_DIR / f"misclassified_{config.EXPERIMENT_NAME}.csv"
HIGH_RISK_CSV_PATH = config.REPORT_DIR / f"high_risk_tumor_to_notumor_{config.EXPERIMENT_NAME}.csv"
MEDICAL_RISK_REPORT_PATH = config.REPORT_DIR / f"medical_risk_report_{config.EXPERIMENT_NAME}.txt"
SAFETY_GATE_REPORT_PATH = config.REPORT_DIR / f"safety_gate_report_{config.EXPERIMENT_NAME}.txt"
SAFETY_GATE_FLAGGED_CSV_PATH = config.REPORT_DIR / f"safety_gate_flagged_{config.EXPERIMENT_NAME}.csv"


@dataclass(frozen=True)
class PredictionRecord:
    image_path: Path
    true_idx: int
    predicted_idx: int
    confidence: float
    probabilities: list[float]

    @property
    def is_correct(self) -> bool:
        return self.true_idx == self.predicted_idx

    def tumor_prob(self, class_names: list[str]) -> float:
        return sum(
            probability
            for class_name, probability in zip(class_names, self.probabilities)
            if class_name != "notumor"
        )

    def notumor_prob(self, class_names: list[str]) -> float:
        notumor_idx = class_names.index("notumor")
        return self.probabilities[notumor_idx]


def load_checkpoint(path: str | None = None) -> dict:
    checkpoint_path = BEST_MODEL_PATH if path is None else config.PROJECT_ROOT / path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def get_sample_paths(loader: torch.utils.data.DataLoader) -> list[Path]:
    dataset = loader.dataset
    if hasattr(dataset, "samples"):
        return [Path(sample[0]) for sample in dataset.samples]
    raise TypeError(
        "Could not infer sample paths from test dataset. "
        "Medical-risk CSV export requires a dataset with a samples attribute."
    )


def predict_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[PredictionRecord]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    records: list[PredictionRecord] = []
    sample_paths = get_sample_paths(loader)
    sample_offset = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="test"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu()
            confidence_values, predicted_indices = probabilities.max(dim=1)
            predictions = predicted_indices.tolist()
            label_values = labels.tolist()

            for batch_index, (true_idx, predicted_idx, confidence, probs) in enumerate(
                zip(
                    label_values,
                    predictions,
                    confidence_values.tolist(),
                    probabilities.tolist(),
                )
            ):
                records.append(
                    PredictionRecord(
                        image_path=sample_paths[sample_offset + batch_index],
                        true_idx=true_idx,
                        predicted_idx=predicted_idx,
                        confidence=confidence,
                        probabilities=probs,
                    )
                )

            sample_offset += len(label_values)
            all_predictions.extend(predictions)
            all_labels.extend(label_values)

    return all_labels, all_predictions, records


def risk_type(record: PredictionRecord, class_names: list[str]) -> str:
    true_class = class_names[record.true_idx]
    predicted_class = class_names[record.predicted_idx]
    true_is_tumor = true_class != "notumor"
    predicted_is_tumor = predicted_class != "notumor"

    if true_is_tumor and predicted_class == "notumor":
        return "tumor_to_notumor"
    if true_class == "notumor" and predicted_is_tumor:
        return "notumor_to_tumor"
    if true_is_tumor and predicted_is_tumor and true_class != predicted_class:
        return "tumor_to_wrong_tumor"
    return "other"


def safety_gate_triggers(record: PredictionRecord, class_names: list[str]) -> bool:
    predicted_class = class_names[record.predicted_idx]
    return (
        config.ENABLE_SAFETY_GATE
        and predicted_class == "notumor"
        and record.tumor_prob(class_names) >= config.TUMOR_PROB_ALERT_THRESHOLD
    )


def safe_output(record: PredictionRecord, class_names: list[str]) -> str:
    if safety_gate_triggers(record, class_names):
        return "uncertain_tumor_review_recommended"
    return class_names[record.predicted_idx]


def compute_medical_risk_metrics(
    matrix: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    notumor_idx = class_to_idx["notumor"]
    tumor_classes = [class_name for class_name in class_names if class_name != "notumor"]
    tumor_indices = [class_to_idx[class_name] for class_name in tumor_classes]

    per_class_false_negative = {
        class_name: int(matrix[class_idx, :].sum() - matrix[class_idx, class_idx])
        for class_name, class_idx in class_to_idx.items()
    }
    per_class_false_positive = {
        class_name: int(matrix[:, class_idx].sum() - matrix[class_idx, class_idx])
        for class_name, class_idx in class_to_idx.items()
    }

    metrics: dict[str, object] = {
        "tumor_to_notumor_total": int(matrix[tumor_indices, notumor_idx].sum()),
        "glioma_to_notumor": int(matrix[class_to_idx["glioma"], notumor_idx]),
        "meningioma_to_notumor": int(matrix[class_to_idx["meningioma"], notumor_idx]),
        "pituitary_to_notumor": int(matrix[class_to_idx["pituitary"], notumor_idx]),
        "notumor_to_tumor_total": int(matrix[notumor_idx, tumor_indices].sum()),
        "per_class_false_negative": per_class_false_negative,
        "per_class_false_positive": per_class_false_positive,
    }
    return metrics


def record_to_csv_row(
    record: PredictionRecord,
    class_names: list[str],
) -> dict[str, object]:
    try:
        image_path = record.image_path.resolve().relative_to(config.PROJECT_ROOT)
    except ValueError:
        image_path = record.image_path.resolve()

    row: dict[str, object] = {
        "image_path": str(image_path),
        "true_class": class_names[record.true_idx],
        "predicted_class": class_names[record.predicted_idx],
        "confidence": f"{record.confidence:.8f}",
        "risk_type": risk_type(record, class_names),
    }
    for class_name, probability in zip(class_names, record.probabilities):
        row[f"probability_{class_name}"] = f"{probability:.8f}"
    return row


def safety_gate_row(record: PredictionRecord, class_names: list[str]) -> dict[str, object]:
    try:
        image_path = record.image_path.resolve().relative_to(config.PROJECT_ROOT)
    except ValueError:
        image_path = record.image_path.resolve()

    row: dict[str, object] = {
        "image_path": str(image_path),
        "true_class": class_names[record.true_idx],
        "predicted_class": class_names[record.predicted_idx],
        "safe_output": safe_output(record, class_names),
        "confidence": f"{record.confidence:.8f}",
        "tumor_prob": f"{record.tumor_prob(class_names):.8f}",
        "notumor_prob": f"{record.notumor_prob(class_names):.8f}",
        "risk_type": risk_type(record, class_names),
    }
    for class_name, probability in zip(class_names, record.probabilities):
        row[f"probability_{class_name}"] = f"{probability:.8f}"
    return row


def write_risk_csvs(records: list[PredictionRecord], class_names: list[str]) -> None:
    columns = [
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "probability_glioma",
        "probability_meningioma",
        "probability_notumor",
        "probability_pituitary",
        "risk_type",
    ]
    misclassified_rows = [
        record_to_csv_row(record, class_names)
        for record in records
        if not record.is_correct
    ]
    high_risk_rows = [
        row for row in misclassified_rows if row["risk_type"] == "tumor_to_notumor"
    ]

    for path, rows in (
        (MISCLASSIFIED_CSV_PATH, misclassified_rows),
        (HIGH_RISK_CSV_PATH, high_risk_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def compute_safety_gate_metrics(
    records: list[PredictionRecord],
    class_names: list[str],
) -> dict[str, int]:
    tumor_to_notumor_records = [
        record for record in records if risk_type(record, class_names) == "tumor_to_notumor"
    ]
    notumor_records = [
        record for record in records if class_names[record.true_idx] == "notumor"
    ]
    flagged_records = [
        record for record in records if safety_gate_triggers(record, class_names)
    ]

    caught_tumor_to_notumor = [
        record for record in tumor_to_notumor_records if safety_gate_triggers(record, class_names)
    ]
    unnecessary_notumor_flags = [
        record for record in notumor_records if safety_gate_triggers(record, class_names)
    ]

    return {
        "safety_gate_enabled": int(config.ENABLE_SAFETY_GATE),
        "tumor_prob_alert_threshold": int(config.TUMOR_PROB_ALERT_THRESHOLD * 1000000),
        "total_flagged": len(flagged_records),
        "tumor_to_notumor_total": len(tumor_to_notumor_records),
        "tumor_to_notumor_caught": len(caught_tumor_to_notumor),
        "tumor_to_notumor_missed": len(tumor_to_notumor_records)
        - len(caught_tumor_to_notumor),
        "notumor_total": len(notumor_records),
        "notumor_flagged_unnecessarily": len(unnecessary_notumor_flags),
    }


def write_safety_gate_outputs(
    records: list[PredictionRecord],
    class_names: list[str],
) -> str:
    flagged_records = [
        record for record in records if safety_gate_triggers(record, class_names)
    ]
    columns = [
        "image_path",
        "true_class",
        "predicted_class",
        "safe_output",
        "confidence",
        "tumor_prob",
        "notumor_prob",
        "probability_glioma",
        "probability_meningioma",
        "probability_notumor",
        "probability_pituitary",
        "risk_type",
    ]
    SAFETY_GATE_FLAGGED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAFETY_GATE_FLAGGED_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([safety_gate_row(record, class_names) for record in flagged_records])

    metrics = compute_safety_gate_metrics(records, class_names)
    threshold = metrics["tumor_prob_alert_threshold"] / 1000000
    report = "\n".join(
        [
            "Safety Gate Report",
            f"Checkpoint: {BEST_MODEL_PATH}",
            f"ENABLE_SAFETY_GATE: {bool(metrics['safety_gate_enabled'])}",
            f"TUMOR_PROB_ALERT_THRESHOLD: {threshold:.4f}",
            "",
            f"total_flagged: {metrics['total_flagged']}",
            f"tumor_to_notumor_total: {metrics['tumor_to_notumor_total']}",
            f"tumor_to_notumor_caught: {metrics['tumor_to_notumor_caught']}",
            f"tumor_to_notumor_missed: {metrics['tumor_to_notumor_missed']}",
            f"notumor_total: {metrics['notumor_total']}",
            f"notumor_flagged_unnecessarily: {metrics['notumor_flagged_unnecessarily']}",
            "",
            "Gate rule:",
            "  If predicted_class == notumor and tumor_prob >= threshold,",
            "  safe_output = uncertain_tumor_review_recommended.",
            "",
            "Safety note:",
            "  This gate reduces decisive no-tumor outputs for suspicious cases.",
            "  It does not clinically validate the model.",
        ]
    )
    SAFETY_GATE_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    return report


def format_medical_risk_report(metrics: dict[str, object]) -> str:
    false_negative = metrics["per_class_false_negative"]
    false_positive = metrics["per_class_false_positive"]
    assert isinstance(false_negative, dict)
    assert isinstance(false_positive, dict)

    lines = [
        "Medical Risk Evaluation Report",
        f"Checkpoint: {BEST_MODEL_PATH}",
        "",
        f"tumor_to_notumor_total: {metrics['tumor_to_notumor_total']}",
        f"glioma_to_notumor: {metrics['glioma_to_notumor']}",
        f"meningioma_to_notumor: {metrics['meningioma_to_notumor']}",
        f"pituitary_to_notumor: {metrics['pituitary_to_notumor']}",
        f"notumor_to_tumor_total: {metrics['notumor_to_tumor_total']}",
        "",
        "Per-class false negative count:",
    ]
    for class_name in config.CLASS_NAMES:
        lines.append(f"  {class_name}: {false_negative[class_name]}")

    lines.append("")
    lines.append("Per-class false positive count:")
    for class_name in config.CLASS_NAMES:
        lines.append(f"  {class_name}: {false_positive[class_name]}")

    lines.extend(
        [
            "",
            "Safety note:",
            "  Tumor-to-notumor predictions are the highest-risk failure mode.",
            "  These outputs are for research/demo purposes only and are not a medical diagnosis.",
        ]
    )
    return "\n".join(lines)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

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
    print(f"Experiment: {config.EXPERIMENT_NAME}")
    print(f"Model: {config.MODEL_NAME}")

    _, _, test_loader, dataset_class_names = create_dataloaders()
    checkpoint = load_checkpoint()
    class_names = checkpoint.get("class_names", dataset_class_names)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels, predictions, prediction_records = predict_loader(model, test_loader, device)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )
    matrix = confusion_matrix(labels, predictions)
    medical_risk_metrics = compute_medical_risk_metrics(matrix, class_names)

    report_text = "\n".join(
        [
            "Test Evaluation Report",
            f"Checkpoint: {BEST_MODEL_PATH}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro F1: {macro_f1:.4f}",
            f"Weighted F1: {weighted_f1:.4f}",
            "",
            report,
            "Confusion matrix:",
            str(matrix),
            "",
            "Medical-risk metrics:",
            f"tumor_to_notumor_total: {medical_risk_metrics['tumor_to_notumor_total']}",
            f"glioma_to_notumor: {medical_risk_metrics['glioma_to_notumor']}",
            f"meningioma_to_notumor: {medical_risk_metrics['meningioma_to_notumor']}",
            f"pituitary_to_notumor: {medical_risk_metrics['pituitary_to_notumor']}",
            f"notumor_to_tumor_total: {medical_risk_metrics['notumor_to_tumor_total']}",
        ]
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    medical_risk_report = format_medical_risk_report(medical_risk_metrics)
    MEDICAL_RISK_REPORT_PATH.write_text(medical_risk_report + "\n", encoding="utf-8")
    write_risk_csvs(prediction_records, class_names)
    safety_gate_report = write_safety_gate_outputs(prediction_records, class_names)
    plot_confusion_matrix(matrix, class_names)

    print(report_text)
    print()
    print(medical_risk_report)
    print()
    print(safety_gate_report)
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Medical risk report saved to: {MEDICAL_RISK_REPORT_PATH}")
    print(f"Misclassified CSV saved to: {MISCLASSIFIED_CSV_PATH}")
    print(f"High-risk CSV saved to: {HIGH_RISK_CSV_PATH}")
    print(f"Safety gate report saved to: {SAFETY_GATE_REPORT_PATH}")
    print(f"Safety gate flagged CSV saved to: {SAFETY_GATE_FLAGGED_CSV_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
