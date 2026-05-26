from __future__ import annotations

import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config
from binary_dataset import get_binary_eval_transform
from dataset import get_eval_transform
from model import build_model
from utils import ensure_dirs, get_device


FOUR_CLASS_CHECKPOINT = config.CHECKPOINT_DIR / "best_model_manifest.pth"
BINARY_CHECKPOINT = config.CHECKPOINT_DIR / "best_binary_tumor_gate.pth"
MISSED_CSV_PATH = config.REPORT_DIR / "combined_gate_missed_high_risk.csv"
SAMPLES_DIR = config.OUTPUT_DIR / "combined_gate_missed_samples"
GRADCAM_DIR = config.OUTPUT_DIR / "gradcam" / "combined_gate_missed"
REPORT_PATH = config.REPORT_DIR / "combined_gate_missed_debug_report.txt"


@dataclass(frozen=True)
class TestRow:
    image_path: Path
    true_class: str


@dataclass(frozen=True)
class MissedSample:
    image_path: Path
    true_class: str
    four_class_pred: str
    four_class_confidence: float
    binary_pred: str
    binary_confidence: float
    four_probs: list[float]
    binary_probs: list[float]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, image_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        logits[:, target_class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return np.zeros_like(cam, dtype=np.float32)
        return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}. {hint}")


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "conv_head"):
        return model.conv_head
    raise AttributeError("Could not find EfficientNet conv_head target layer.")


def load_model(
    checkpoint_path: Path,
    default_class_names: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    require_file(checkpoint_path, "Train/evaluate the required model first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names", default_class_names)
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def read_test_rows() -> list[TestRow]:
    require_file(
        config.BINARY_TEST_MANIFEST,
        "Run `python src\\create_binary_manifest.py` first.",
    )
    rows: list[TestRow] = []
    with config.BINARY_TEST_MANIFEST.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                TestRow(
                    image_path=config.PROJECT_ROOT / row["path"],
                    true_class=row["original_class_name"],
                )
            )
    return rows


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


def find_missed_samples(
    rows: list[TestRow],
    four_model: torch.nn.Module,
    four_classes: list[str],
    binary_model: torch.nn.Module,
    binary_classes: list[str],
    device: torch.device,
) -> list[MissedSample]:
    four_transform = get_eval_transform()
    binary_transform = get_binary_eval_transform()
    missed: list[MissedSample] = []

    for row in tqdm(rows, desc="combined-gate-misses"):
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

        if is_tumor(row.true_class) and four_pred == "notumor" and binary_pred == "notumor":
            missed.append(
                MissedSample(
                    image_path=row.image_path,
                    true_class=row.true_class,
                    four_class_pred=four_pred,
                    four_class_confidence=four_conf,
                    binary_pred=binary_pred,
                    binary_confidence=binary_conf,
                    four_probs=four_probs,
                    binary_probs=binary_probs,
                )
            )

    return missed


def safe_stem(sample: MissedSample) -> str:
    return (
        f"{sample.true_class}__4class-{sample.four_class_pred}"
        f"__4conf-{sample.four_class_confidence:.4f}"
        f"__bin-{sample.binary_pred}"
        f"__binconf-{sample.binary_confidence:.4f}"
        f"__{sample.image_path.stem}"
    )


def copy_missed_samples(samples: list[MissedSample]) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        destination = SAMPLES_DIR / f"{safe_stem(sample)}{sample.image_path.suffix}"
        shutil.copy2(sample.image_path, destination)


def write_missed_csv(
    samples: list[MissedSample],
    four_classes: list[str],
    binary_classes: list[str],
) -> None:
    columns = [
        "image_path",
        "true_class",
        "four_class_pred",
        "four_class_confidence",
        "binary_pred",
        "binary_confidence",
        "probability_glioma",
        "probability_meningioma",
        "probability_notumor",
        "probability_pituitary",
        "binary_probability_tumor",
        "binary_probability_notumor",
        "risk_type",
    ]
    MISSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MISSED_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_path": str(sample.image_path.resolve().relative_to(config.PROJECT_ROOT)),
                    "true_class": sample.true_class,
                    "four_class_pred": sample.four_class_pred,
                    "four_class_confidence": f"{sample.four_class_confidence:.8f}",
                    "binary_pred": sample.binary_pred,
                    "binary_confidence": f"{sample.binary_confidence:.8f}",
                    "probability_glioma": f"{sample.four_probs[four_classes.index('glioma')]:.8f}",
                    "probability_meningioma": f"{sample.four_probs[four_classes.index('meningioma')]:.8f}",
                    "probability_notumor": f"{sample.four_probs[four_classes.index('notumor')]:.8f}",
                    "probability_pituitary": f"{sample.four_probs[four_classes.index('pituitary')]:.8f}",
                    "binary_probability_tumor": f"{sample.binary_probs[binary_classes.index('tumor')]:.8f}",
                    "binary_probability_notumor": f"{sample.binary_probs[binary_classes.index('notumor')]:.8f}",
                    "risk_type": "combined_gate_missed_tumor_to_notumor",
                }
            )


def overlay_cam(image: Image.Image, cam_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    base_array = np.asarray(base).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam_map)[..., :3].astype(np.float32)
    overlay = ((1 - alpha) * base_array + alpha * heatmap).clip(0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def generate_gradcams(
    samples: list[MissedSample],
    four_model: torch.nn.Module,
    four_classes: list[str],
    binary_model: torch.nn.Module,
    binary_classes: list[str],
    device: torch.device,
) -> list[str]:
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    four_transform = get_eval_transform()
    binary_transform = get_binary_eval_transform()
    four_cam = GradCAM(four_model, get_target_layer(four_model))
    binary_cam = GradCAM(binary_model, get_target_layer(binary_model))
    failures: list[str] = []

    try:
        for sample in tqdm(samples, desc="gradcam-misses"):
            try:
                stem = safe_stem(sample)
                with Image.open(sample.image_path) as image:
                    image = image.convert("RGB")
                    original_output = GRADCAM_DIR / f"{stem}__original{sample.image_path.suffix}"
                    shutil.copy2(sample.image_path, original_output)

                    four_tensor = four_transform(image).unsqueeze(0).to(device)
                    binary_tensor = binary_transform(image).unsqueeze(0).to(device)

                    four_pred_cam = four_cam.generate(
                        four_tensor,
                        four_classes.index("notumor"),
                    )
                    overlay_cam(image, four_pred_cam).save(
                        GRADCAM_DIR / f"{stem}__4class-gradcam-pred-notumor.png"
                    )

                    four_true_cam = four_cam.generate(
                        four_tensor,
                        four_classes.index(sample.true_class),
                    )
                    overlay_cam(image, four_true_cam).save(
                        GRADCAM_DIR / f"{stem}__4class-gradcam-true-{sample.true_class}.png"
                    )

                    binary_pred_cam = binary_cam.generate(
                        binary_tensor,
                        binary_classes.index("notumor"),
                    )
                    overlay_cam(image, binary_pred_cam).save(
                        GRADCAM_DIR / f"{stem}__binary-gradcam-pred-notumor.png"
                    )

                    binary_true_cam = binary_cam.generate(
                        binary_tensor,
                        binary_classes.index("tumor"),
                    )
                    overlay_cam(image, binary_true_cam).save(
                        GRADCAM_DIR / f"{stem}__binary-gradcam-true-tumor.png"
                    )
            except Exception as exc:
                failures.append(f"{sample.image_path}: {exc}")
    finally:
        four_cam.close()
        binary_cam.close()

    return failures


def write_report(samples: list[MissedSample], gradcam_failures: list[str]) -> None:
    counts = Counter(sample.true_class for sample in samples)
    lines = [
        "Combined Gate Missed High-Risk Debug Report",
        f"4-class checkpoint: {FOUR_CLASS_CHECKPOINT}",
        f"Binary checkpoint: {BINARY_CHECKPOINT}",
        f"Total missed high-risk samples: {len(samples)}",
        "",
        "Count by true class:",
    ]
    for class_name in ["glioma", "meningioma", "pituitary"]:
        lines.append(f"  {class_name}: {counts[class_name]}")

    lines.extend(
        [
            "",
            "Output folders:",
            f"  Missed samples: {SAMPLES_DIR}",
            f"  Grad-CAM: {GRADCAM_DIR}",
            f"  CSV: {MISSED_CSV_PATH}",
            "",
            f"Grad-CAM failed samples: {len(gradcam_failures)}",
            "",
            "Interpretation note:",
            "  Grad-CAM is qualitative debugging evidence only, not clinical evidence.",
            "  These outputs do not validate clinical safety.",
            "  Raw dataset files were not modified, moved, deleted, or renamed.",
        ]
    )
    if gradcam_failures:
        lines.extend(["", "Grad-CAM failures:", *[f"  {item}" for item in gradcam_failures]])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ensure_dirs()
    require_file(
        config.BINARY_TEST_MANIFEST,
        "Run `python src\\create_binary_manifest.py` first.",
    )
    require_file(
        config.REPORT_DIR / "combined_gate_flagged.csv",
        "Run `python src\\evaluate_combined_gate.py` first. This file is used as a run artifact check.",
    )

    device = get_device()
    print(f"Using device: {device}")
    four_model, four_classes = load_model(FOUR_CLASS_CHECKPOINT, config.CLASS_NAMES, device)
    binary_model, binary_classes = load_model(
        BINARY_CHECKPOINT,
        config.BINARY_CLASS_NAMES,
        device,
    )

    rows = read_test_rows()
    missed_samples = find_missed_samples(
        rows,
        four_model,
        four_classes,
        binary_model,
        binary_classes,
        device,
    )
    write_missed_csv(missed_samples, four_classes, binary_classes)
    copy_missed_samples(missed_samples)
    gradcam_failures = generate_gradcams(
        missed_samples,
        four_model,
        four_classes,
        binary_model,
        binary_classes,
        device,
    )
    write_report(missed_samples, gradcam_failures)

    return 1 if gradcam_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
