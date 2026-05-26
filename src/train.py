from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import build_model
from utils import ensure_dirs, get_device, set_seed


BEST_MODEL_PATH = config.CHECKPOINT_DIR / f"best_model_{config.EXPERIMENT_NAME}.pth"
LAST_MODEL_PATH = config.CHECKPOINT_DIR / f"last_model_{config.EXPERIMENT_NAME}.pth"
CURVES_PATH = config.OUTPUT_DIR / f"training_curves_{config.EXPERIMENT_NAME}.png"


def get_dataset_targets(dataset: torch.utils.data.Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]

    if isinstance(dataset, torch.utils.data.Subset):
        parent_dataset = dataset.dataset
        if hasattr(parent_dataset, "targets"):
            return [int(parent_dataset.targets[index]) for index in dataset.indices]

    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]

    raise TypeError(
        "Could not infer targets from training dataset. "
        "Expected a dataset with targets, samples, or a Subset wrapping one."
    )


def compute_class_weights(
    train_dataset: torch.utils.data.Dataset,
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    targets = get_dataset_targets(train_dataset)
    counts = Counter(targets)
    num_classes = len(class_names)
    total = len(targets)

    missing_classes = [
        class_names[class_idx]
        for class_idx in range(num_classes)
        if counts[class_idx] == 0
    ]
    if missing_classes:
        raise ValueError(f"Cannot compute class weights; missing classes: {missing_classes}")

    weights = [
        total / (num_classes * counts[class_idx])
        for class_idx in range(num_classes)
    ]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training class counts used for loss weighting:")
    for class_idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {counts[class_idx]}")

    print("CrossEntropyLoss class weights:")
    for class_name, weight in zip(class_names, class_weights.detach().cpu().tolist()):
        print(f"  {class_name}: {weight:.6f}")

    return class_weights


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    progress = tqdm(loader, leave=False, desc="train" if is_training else "eval")
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    val_loss: float,
    val_acc: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "config": config.as_dict(),
    }
    torch.save(checkpoint, path)


def plot_training_curves(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    set_seed(config.RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Experiment: {config.EXPERIMENT_NAME}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Pad to square: {config.USE_PAD_TO_SQUARE}")
    print(f"Horizontal flip: {config.USE_HORIZONTAL_FLIP}")
    print(f"CLAHE: {config.USE_CLAHE}")

    train_loader, val_loader, _, class_names = create_dataloaders()
    model = build_model(num_classes=len(class_names)).to(device)

    if config.USE_MANIFEST:
        class_weights = compute_class_weights(train_loader.dataset, class_names, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("USE_MANIFEST=False; using unweighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = 5

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        save_checkpoint(
            LAST_MODEL_PATH,
            epoch,
            model,
            optimizer,
            class_names,
            val_loss,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                model,
                optimizer,
                class_names,
                val_loss,
                val_acc,
            )
            print(f"Saved new best checkpoint: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"No val_loss improvement for {epochs_without_improvement} epoch(s).")

        plot_training_curves(history)

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Last checkpoint: {LAST_MODEL_PATH}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Training curves: {CURVES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
