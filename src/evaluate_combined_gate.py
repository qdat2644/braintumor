from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import config
from binary_dataset import get_binary_eval_transform
from dataset import get_eval_transform
from model import build_model
from utils import ensure_dirs, get_device


FOUR_CLASS_CHECKPOINT = config.CHECKPOINT_DIR / "best_model_manifest.pth"
BINARY_CHECKPOINT = config.CHECKPOINT_DIR / "best_binary_tumor_gate.pth"
REPORT_PATH = config.REPORT_DIR / "combined_gate_report.txt"
FLAGGED_CSV_PATH = config.REPORT_DIR / "combined_gate_flagged.csv"
TE_GL_143_PATH = config.PROJECT_ROOT / "data" / "extracted" / "Testing" / "glioma" / "Te-gl_143.jpg"


@dataclass(frozen=True)
class TestRow:
    image_path: Path
    original_class_name: str


def read_binary_test_manifest() -> list[TestRow]:
    if not config.BINARY_TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Binary test manifest not found: {config.BINARY_TEST_MANIFEST}. "
            "Run `python src\\create_binary_manifest.py` first."
        )

    rows: list[TestRow] = []
    with config.BINARY_TEST_MANIFEST.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                TestRow(
                    image_path=config.PROJECT_ROOT / row["path"],
                    original_class_name=row["original_class_name"],
                )
            )
    return rows


def load_checkpoint_model(
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


def predict_image(
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


def is_tumor(class_name: str) -> bool:
    return class_name != "notumor"


def main() -> int:
    ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    four_model, four_classes = load_checkpoint_model(
        FOUR_CLASS_CHECKPOINT,
        config.CLASS_NAMES,
        device,
    )
    binary_model, binary_classes = load_checkpoint_model(
        BINARY_CHECKPOINT,
        config.BINARY_CLASS_NAMES,
        device,
    )

    four_transform = get_eval_transform()
    binary_transform = get_binary_eval_transform()
    rows = read_binary_test_manifest()

    original_tumor_to_notumor = 0
    caught = 0
    missed = 0
    notumor_flagged_unnecessarily = 0
    te_gl_143_caught = False
    flagged_rows: list[dict[str, object]] = []

    for row in tqdm(rows, desc="combined-gate"):
        four_pred, four_conf, four_probs = predict_image(
            four_model,
            four_transform,
            row.image_path,
            four_classes,
            device,
        )
        binary_pred, binary_conf, binary_probs = predict_image(
            binary_model,
            binary_transform,
            row.image_path,
            binary_classes,
            device,
        )

        gate_triggers = four_pred == "notumor" and binary_pred == "tumor"
        safe_output = "uncertain_tumor_review_recommended" if gate_triggers else four_pred
        true_is_tumor = is_tumor(row.original_class_name)

        if true_is_tumor and four_pred == "notumor":
            original_tumor_to_notumor += 1
            if gate_triggers:
                caught += 1
            else:
                missed += 1

        if row.original_class_name == "notumor" and gate_triggers:
            notumor_flagged_unnecessarily += 1

        if row.image_path.resolve() == TE_GL_143_PATH.resolve() and gate_triggers:
            te_gl_143_caught = True

        if gate_triggers:
            flagged_rows.append(
                {
                    "image_path": str(row.image_path.resolve().relative_to(config.PROJECT_ROOT)),
                    "true_class": row.original_class_name,
                    "four_class_prediction": four_pred,
                    "four_class_confidence": f"{four_conf:.8f}",
                    "binary_prediction": binary_pred,
                    "binary_confidence": f"{binary_conf:.8f}",
                    "safe_output": safe_output,
                    "four_prob_glioma": f"{four_probs[four_classes.index('glioma')]:.8f}",
                    "four_prob_meningioma": f"{four_probs[four_classes.index('meningioma')]:.8f}",
                    "four_prob_notumor": f"{four_probs[four_classes.index('notumor')]:.8f}",
                    "four_prob_pituitary": f"{four_probs[four_classes.index('pituitary')]:.8f}",
                    "binary_prob_tumor": f"{binary_probs[binary_classes.index('tumor')]:.8f}",
                    "binary_prob_notumor": f"{binary_probs[binary_classes.index('notumor')]:.8f}",
                }
            )

    FLAGGED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_path",
        "true_class",
        "four_class_prediction",
        "four_class_confidence",
        "binary_prediction",
        "binary_confidence",
        "safe_output",
        "four_prob_glioma",
        "four_prob_meningioma",
        "four_prob_notumor",
        "four_prob_pituitary",
        "binary_prob_tumor",
        "binary_prob_notumor",
    ]
    with FLAGGED_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flagged_rows)

    report_text = "\n".join(
        [
            "Combined Binary Tumor Gate Report",
            f"4-class checkpoint: {FOUR_CLASS_CHECKPOINT}",
            f"Binary checkpoint: {BINARY_CHECKPOINT}",
            "",
            f"original_4class_tumor_to_notumor_total: {original_tumor_to_notumor}",
            f"combined_gate_caught_count: {caught}",
            f"combined_gate_missed_count: {missed}",
            f"notumor_flagged_unnecessarily: {notumor_flagged_unnecessarily}",
            f"total_flagged_by_combined_gate: {len(flagged_rows)}",
            f"te_gl_143_caught: {te_gl_143_caught}",
            "",
            "Gate logic:",
            "  If 4-class predicts notumor but binary gate predicts tumor,",
            "  safe_output = uncertain_tumor_review_recommended.",
            "",
            "Safety note: This is an engineering safety gate, not clinical validation.",
        ]
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    print(f"Combined gate report saved to: {REPORT_PATH}")
    print(f"Combined gate flagged CSV saved to: {FLAGGED_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
