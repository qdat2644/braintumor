from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from dataset import IMAGENET_MEAN, IMAGENET_STD


def get_binary_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_binary_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BinaryManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, transform: transforms.Compose) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Binary manifest not found: {manifest_path}. "
                "Run `python src\\create_binary_manifest.py` first."
            )

        self.manifest_path = manifest_path
        self.transform = transform
        self.classes = list(config.BINARY_CLASS_NAMES)
        self.samples: list[tuple[Path, int]] = []
        self.targets: list[int] = []
        self.original_class_names: list[str] = []

        with manifest_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required_columns = {
                "path",
                "split",
                "original_class_name",
                "binary_class_name",
                "binary_class_idx",
                "md5",
                "width",
                "height",
            }
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{manifest_path} is missing columns: {sorted(missing_columns)}"
                )

            for row in reader:
                class_idx = int(row["binary_class_idx"])
                class_name = row["binary_class_name"]
                if class_name != self.classes[class_idx]:
                    raise ValueError(f"Inconsistent binary class mapping: {row}")

                image_path = config.PROJECT_ROOT / row["path"]
                self.samples.append((image_path, class_idx))
                self.targets.append(class_idx)
                self.original_class_names.append(row["original_class_name"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


def _print_distribution(name: str, dataset: BinaryManifestDataset) -> None:
    counts = Counter(dataset.targets)
    readable = {
        dataset.classes[class_idx]: counts[class_idx]
        for class_idx in sorted(counts)
    }
    print(f"{name} distribution: {readable}")


def create_binary_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dataset = BinaryManifestDataset(
        config.BINARY_TRAIN_MANIFEST,
        transform=get_binary_train_transform(),
    )
    val_dataset = BinaryManifestDataset(
        config.BINARY_VAL_MANIFEST,
        transform=get_binary_eval_transform(),
    )
    test_dataset = BinaryManifestDataset(
        config.BINARY_TEST_MANIFEST,
        transform=get_binary_eval_transform(),
    )

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    _print_distribution("Binary train", train_dataset)
    _print_distribution("Binary val", val_dataset)
    _print_distribution("Binary test", test_dataset)

    return train_loader, val_loader, test_loader, train_dataset.classes
