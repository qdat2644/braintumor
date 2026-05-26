from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dataset import get_eval_transform
from model import build_model
from utils import get_device


CHECKPOINT_PATH = config.CHECKPOINT_DIR / "best_model_manifest.pth"
HIGH_RISK_CSV = config.REPORT_DIR / "high_risk_tumor_to_notumor_manifest.csv"
OUTPUT_DIR = config.OUTPUT_DIR / "gradcam" / "high_risk_tumor_to_notumor"
REPORT_PATH = config.REPORT_DIR / "gradcam_high_risk_report.txt"


@dataclass(frozen=True)
class HighRiskRow:
    image_path: Path
    true_class: str
    predicted_class: str
    confidence: float


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
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
        score = logits[:, target_class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
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


def read_high_risk_rows(path: Path) -> list[HighRiskRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"High-risk CSV not found: {path}. Run `python src\\evaluate.py` first."
        )

    rows: list[HighRiskRow] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                HighRiskRow(
                    image_path=config.PROJECT_ROOT / row["image_path"],
                    true_class=row["true_class"],
                    predicted_class=row["predicted_class"],
                    confidence=float(row["confidence"]),
                )
            )
    return rows


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "conv_head"):
        return model.conv_head
    raise AttributeError("Could not find EfficientNet conv_head target layer for Grad-CAM.")


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def safe_stem(row: HighRiskRow) -> str:
    return (
        f"{row.true_class}__pred-{row.predicted_class}"
        f"__conf-{row.confidence:.4f}__{row.image_path.stem}"
    )


def overlay_cam(image: Image.Image, cam_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    base_array = np.asarray(base).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam_map)[..., :3].astype(np.float32)
    overlay = ((1 - alpha) * base_array + alpha * heatmap).clip(0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def save_original(image_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination)


def main() -> int:
    rows = read_high_risk_rows(HIGH_RISK_CSV)
    device = get_device()
    model, class_names = load_model(device)
    transform = get_eval_transform()
    gradcam = GradCAM(model, get_target_layer(model))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed: list[str] = []

    try:
        for row in rows:
            try:
                if not row.image_path.exists():
                    raise FileNotFoundError(row.image_path)

                true_idx = class_names.index(row.true_class)
                predicted_idx = class_names.index(row.predicted_class)
                stem = safe_stem(row)

                with Image.open(row.image_path) as image:
                    image = image.convert("RGB")
                    input_tensor = transform(image).unsqueeze(0).to(device)

                    original_output = OUTPUT_DIR / f"{stem}__original{row.image_path.suffix}"
                    save_original(row.image_path, original_output)

                    pred_cam = gradcam.generate(input_tensor, predicted_idx)
                    pred_overlay = overlay_cam(image, pred_cam)
                    pred_overlay.save(
                        OUTPUT_DIR / f"{stem}__gradcam-pred-{row.predicted_class}.png"
                    )

                    true_cam = gradcam.generate(input_tensor, true_idx)
                    true_overlay = overlay_cam(image, true_cam)
                    true_overlay.save(
                        OUTPUT_DIR / f"{stem}__gradcam-true-{row.true_class}.png"
                    )

                processed += 1
            except Exception as exc:
                failed.append(f"{row.image_path}: {exc}")
    finally:
        gradcam.close()

    report_lines = [
        "Grad-CAM High-Risk False Negative Report",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Input CSV: {HIGH_RISK_CSV}",
        f"Output folder: {OUTPUT_DIR}",
        f"Processed high-risk samples: {processed}",
        f"Failed samples: {len(failed)}",
        "",
        "Generated outputs per processed sample:",
        "  original image copy",
        "  Grad-CAM overlay for predicted class",
        "  Grad-CAM overlay for true class",
        "",
        "Interpretation note:",
        "  Grad-CAM is qualitative debugging evidence only, not clinical evidence.",
        "  It does not prove the model used medically valid anatomy.",
    ]
    if failed:
        report_lines.extend(["", "Failed samples:", *[f"  {item}" for item in failed]])

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
