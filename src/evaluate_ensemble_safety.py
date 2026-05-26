from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

import config
from dataset import IMAGENET_MEAN, IMAGENET_STD
from utils import ensure_dirs, get_device


FOUR_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}

TEST_MANIFEST = config.PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"
HARD_CASE_CSV = config.REPORT_DIR / "combined_gate_missed_high_risk.csv"

REPORT_PATH = config.REPORT_DIR / "ensemble_safety_report.txt"
PREDICTIONS_CSV_PATH = config.REPORT_DIR / "ensemble_safety_predictions.csv"

TARGET_HARD_CASES = {
    "data/extracted/Testing/glioma/Te-gl_143.jpg": "Te-gl_143",
    "data/extracted/Testing/glioma/Te-gl_341.jpg": "Te-gl_341",
    "data/extracted/Testing/glioma/Te-gl_74.jpg": "Te-gl_74",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint_path: Path
    model_name: str
    image_size: int
    use_pad_to_square: bool


@dataclass(frozen=True)
class SampleRow:
    image_path: Path
    relative_path: str
    true_class: str
    is_hard_case: bool


@dataclass(frozen=True)
class ModelPrediction:
    predicted_class: str
    confidence: float
    probabilities: list[float]

    @property
    def predicts_tumor(self) -> bool:
        return self.predicted_class in TUMOR_CLASSES

    @property
    def tumor_prob(self) -> float:
        return sum(
            probability
            for class_name, probability in zip(FOUR_CLASS_NAMES, self.probabilities)
            if class_name in TUMOR_CLASSES
        )

    @property
    def notumor_prob(self) -> float:
        return self.probabilities[FOUR_CLASS_NAMES.index("notumor")]


@dataclass(frozen=True)
class EnsembleRecord:
    sample: SampleRow
    efficientnet: ModelPrediction
    densenet: ModelPrediction
    convnext: ModelPrediction


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


MODEL_SPECS = [
    ModelSpec(
        key="efficientnet",
        checkpoint_path=config.CHECKPOINT_DIR / "best_model_manifest.pth",
        model_name="efficientnet_b0",
        image_size=224,
        use_pad_to_square=False,
    ),
    ModelSpec(
        key="densenet",
        checkpoint_path=config.CHECKPOINT_DIR / "best_model_densenet121_pad224_nohflip.pth",
        model_name="densenet121",
        image_size=224,
        use_pad_to_square=True,
    ),
    ModelSpec(
        key="convnext",
        checkpoint_path=config.CHECKPOINT_DIR / "best_model_convnext_tiny_pad224_nohflip.pth",
        model_name="convnext_tiny",
        image_size=224,
        use_pad_to_square=True,
    ),
]


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {purpose}: {path}")


def normalize_relative_path(path_text: str) -> str:
    return Path(path_text).as_posix()


def build_transform(spec: ModelSpec) -> transforms.Compose:
    steps = []
    if spec.use_pad_to_square:
        steps.append(PadToSquare())
    steps.extend(
        [
            transforms.Resize((spec.image_size, spec.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    require_file(spec.checkpoint_path, f"{spec.key} checkpoint")
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names", FOUR_CLASS_NAMES)
    if class_names != FOUR_CLASS_NAMES:
        raise ValueError(
            f"{spec.key} class order mismatch. Found {class_names}, expected {FOUR_CLASS_NAMES}."
        )

    model = timm.create_model(
        spec.model_name,
        pretrained=False,
        num_classes=len(FOUR_CLASS_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def read_hard_case_paths() -> set[str]:
    require_file(HARD_CASE_CSV, "hard-case evaluation")
    paths: set[str] = set()
    with HARD_CASE_CSV.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"{HARD_CASE_CSV} must contain image_path column.")
        for row in reader:
            paths.add(normalize_relative_path(row["image_path"]))
    return paths


def read_test_samples() -> list[SampleRow]:
    require_file(TEST_MANIFEST, "test-set ensemble evaluation")
    hard_case_paths = read_hard_case_paths()
    rows: list[SampleRow] = []

    with TEST_MANIFEST.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"path", "class_name"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{TEST_MANIFEST} missing columns: {sorted(missing)}")

        for row in reader:
            relative_path = normalize_relative_path(row["path"])
            rows.append(
                SampleRow(
                    image_path=config.PROJECT_ROOT / relative_path,
                    relative_path=relative_path,
                    true_class=row["class_name"],
                    is_hard_case=relative_path in hard_case_paths,
                )
            )

    return rows


def predict_one(
    model: torch.nn.Module,
    transform: transforms.Compose,
    image_path: Path,
    device: torch.device,
) -> ModelPrediction:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, predicted_idx = probabilities.max(dim=0)
    return ModelPrediction(
        predicted_class=FOUR_CLASS_NAMES[predicted_idx.item()],
        confidence=confidence.item(),
        probabilities=probabilities.tolist(),
    )


def is_tumor(class_name: str) -> bool:
    return class_name in TUMOR_CLASSES


def strategy_any_model_tumor(record: EnsembleRecord) -> str:
    if (
        record.efficientnet.predicts_tumor
        or record.densenet.predicts_tumor
        or record.convnext.predicts_tumor
    ):
        return "tumor_or_uncertain"
    return "notumor"


def strategy_two_of_three_tumor(record: EnsembleRecord) -> str:
    votes = sum(
        [
            record.efficientnet.predicts_tumor,
            record.densenet.predicts_tumor,
            record.convnext.predicts_tumor,
        ]
    )
    return "tumor_or_uncertain" if votes >= 2 else "notumor"


def strategy_convnext_or_densenet_tumor(record: EnsembleRecord) -> str:
    if record.convnext.predicts_tumor or record.densenet.predicts_tumor:
        return "tumor_or_uncertain"
    return "notumor"


def strategy_convnext_primary_densenet_override(record: EnsembleRecord) -> str:
    if record.convnext.predicts_tumor:
        return "tumor_or_uncertain"
    if record.convnext.predicted_class == "notumor" and record.densenet.predicts_tumor:
        return "tumor_or_uncertain"
    return "notumor"


STRATEGIES: dict[str, Callable[[EnsembleRecord], str]] = {
    "any_model_tumor": strategy_any_model_tumor,
    "two_of_three_tumor": strategy_two_of_three_tumor,
    "convnext_or_densenet_tumor": strategy_convnext_or_densenet_tumor,
    "convnext_primary_densenet_override": strategy_convnext_primary_densenet_override,
}


def evaluate_strategy(
    strategy_name: str,
    strategy_fn: Callable[[EnsembleRecord], str],
    records: list[EnsembleRecord],
) -> dict[str, object]:
    original_high_risk = [
        record
        for record in records
        if is_tumor(record.sample.true_class)
        and record.efficientnet.predicted_class == "notumor"
    ]
    hard_cases = [record for record in records if record.sample.is_hard_case]

    caught_high_risk = [
        record for record in original_high_risk if strategy_fn(record) != "notumor"
    ]
    missed_high_risk = [
        record for record in original_high_risk if strategy_fn(record) == "notumor"
    ]
    unnecessary_notumor_flags = [
        record
        for record in records
        if record.sample.true_class == "notumor" and strategy_fn(record) != "notumor"
    ]
    hard_recovered = [
        record for record in hard_cases if strategy_fn(record) != "notumor"
    ]
    hard_still_notumor = [
        record for record in hard_cases if strategy_fn(record) == "notumor"
    ]

    target_results = {}
    for target_path, label in TARGET_HARD_CASES.items():
        matched = next(
            (record for record in records if record.sample.relative_path == target_path),
            None,
        )
        target_results[label] = bool(matched and strategy_fn(matched) != "notumor")

    return {
        "strategy": strategy_name,
        "original_4class_tumor_to_notumor_total": len(original_high_risk),
        "tumor_to_notumor_caught_count": len(caught_high_risk),
        "tumor_to_notumor_missed_count": len(missed_high_risk),
        "notumor_flagged_unnecessarily": len(unnecessary_notumor_flags),
        "hard_case_total": len(hard_cases),
        "hard_case_recovered_count": len(hard_recovered),
        "hard_case_still_notumor_count": len(hard_still_notumor),
        **target_results,
    }


def row_for_csv(record: EnsembleRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "image_path": record.sample.relative_path,
        "true_class": record.sample.true_class,
        "is_hard_case": record.sample.is_hard_case,
    }

    for key, prediction in (
        ("efficientnet", record.efficientnet),
        ("densenet", record.densenet),
        ("convnext", record.convnext),
    ):
        row[f"{key}_pred"] = prediction.predicted_class
        row[f"{key}_confidence"] = f"{prediction.confidence:.8f}"
        row[f"{key}_tumor_prob"] = f"{prediction.tumor_prob:.8f}"
        row[f"{key}_notumor_prob"] = f"{prediction.notumor_prob:.8f}"

    for strategy_name, strategy_fn in STRATEGIES.items():
        row[f"{strategy_name}_safe_output"] = strategy_fn(record)

    return row


def write_predictions_csv(records: list[EnsembleRecord]) -> None:
    columns = [
        "image_path",
        "true_class",
        "is_hard_case",
        "efficientnet_pred",
        "efficientnet_confidence",
        "efficientnet_tumor_prob",
        "efficientnet_notumor_prob",
        "densenet_pred",
        "densenet_confidence",
        "densenet_tumor_prob",
        "densenet_notumor_prob",
        "convnext_pred",
        "convnext_confidence",
        "convnext_tumor_prob",
        "convnext_notumor_prob",
        *[f"{strategy_name}_safe_output" for strategy_name in STRATEGIES],
    ]
    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([row_for_csv(record) for record in records])


def write_report(strategy_metrics: list[dict[str, object]], records: list[EnsembleRecord]) -> None:
    lines = [
        "Ensemble Safety Evaluation Report",
        "",
        "Models:",
        f"  EfficientNet baseline: {MODEL_SPECS[0].checkpoint_path}",
        f"  DenseNet121: {MODEL_SPECS[1].checkpoint_path}",
        f"  ConvNeXt-Tiny: {MODEL_SPECS[2].checkpoint_path}",
        "",
        f"Test samples evaluated: {len(records)}",
        f"Hard cases evaluated: {sum(record.sample.is_hard_case for record in records)}",
        "",
        "Strategy metrics:",
    ]

    for metrics in strategy_metrics:
        lines.extend(
            [
                "",
                f"[{metrics['strategy']}]",
                f"  original_4class_tumor_to_notumor_total: {metrics['original_4class_tumor_to_notumor_total']}",
                f"  tumor_to_notumor_caught_count: {metrics['tumor_to_notumor_caught_count']}",
                f"  tumor_to_notumor_missed_count: {metrics['tumor_to_notumor_missed_count']}",
                f"  notumor_flagged_unnecessarily: {metrics['notumor_flagged_unnecessarily']}",
                f"  hard_case_recovered_count: {metrics['hard_case_recovered_count']}",
                f"  hard_case_still_notumor_count: {metrics['hard_case_still_notumor_count']}",
                f"  Te-gl_143 caught: {metrics['Te-gl_143']}",
                f"  Te-gl_341 caught: {metrics['Te-gl_341']}",
                f"  Te-gl_74 caught: {metrics['Te-gl_74']}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation note:",
            "  This is a safety-oriented audit of model disagreement, not clinical validation.",
            "  Tumor-or-uncertain output should trigger review wording, not diagnosis wording.",
            "  Every count above is computed from model predictions generated by this script.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    samples = read_test_samples()
    loaded_models = {
        spec.key: (load_model(spec, device), build_transform(spec))
        for spec in MODEL_SPECS
    }

    records: list[EnsembleRecord] = []
    for sample in tqdm(samples, desc="ensemble-safety"):
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing test image: {sample.image_path}")

        predictions = {
            key: predict_one(model, transform, sample.image_path, device)
            for key, (model, transform) in loaded_models.items()
        }
        records.append(
            EnsembleRecord(
                sample=sample,
                efficientnet=predictions["efficientnet"],
                densenet=predictions["densenet"],
                convnext=predictions["convnext"],
            )
        )

    strategy_metrics = [
        evaluate_strategy(strategy_name, strategy_fn, records)
        for strategy_name, strategy_fn in STRATEGIES.items()
    ]
    write_predictions_csv(records)
    write_report(strategy_metrics, records)

    print(f"Predictions CSV saved to: {PREDICTIONS_CSV_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
