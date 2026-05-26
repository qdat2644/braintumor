from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = max(width, height)
        left = (size - width) // 2
        top = (size - height) // 2
        right = size - width - left
        bottom = size - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "USE_CLAHE=True requires opencv-python. Install only opencv-python if needed; do not reinstall PyTorch."
            ) from exc

        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)


def _base_preprocess_steps() -> list:
    steps = []
    if config.USE_PAD_TO_SQUARE:
        steps.append(PadToSquare())
    if config.USE_CLAHE:
        steps.append(CLAHETransform())
    steps.append(transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    return steps


def get_train_transform() -> transforms.Compose:
    steps = _base_preprocess_steps()
    if config.USE_HORIZONTAL_FLIP:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(
        steps
    )


def get_eval_transform() -> transforms.Compose:
    steps = _base_preprocess_steps()
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(
        steps
    )


def _validate_class_names(class_names: list[str]) -> None:
    if class_names != config.CLASS_NAMES:
        raise ValueError(
            "Class folder order does not match config.CLASS_NAMES. "
            f"Found {class_names}, expected {config.CLASS_NAMES}."
        )


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, transform: transforms.Compose) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run `python src\\create_manifest.py` first, or set USE_MANIFEST=False."
            )

        self.manifest_path = manifest_path
        self.transform = transform
        self.classes = list(config.CLASS_NAMES)
        self.samples: list[tuple[Path, int]] = []
        self.targets: list[int] = []

        with manifest_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required_columns = {
                "path",
                "split",
                "class_name",
                "class_idx",
                "md5",
                "width",
                "height",
            }
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Manifest {manifest_path} is missing columns: {sorted(missing_columns)}"
                )

            for row in reader:
                class_name = row["class_name"]
                class_idx = int(row["class_idx"])
                if class_name != config.CLASS_NAMES[class_idx]:
                    raise ValueError(
                        f"Manifest row has inconsistent class mapping: {row}"
                    )

                image_path = config.PROJECT_ROOT / row["path"]
                self.samples.append((image_path, class_idx))
                self.targets.append(class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def _stratified_train_val_indices(
    targets: list[int], val_size: float = 0.15
) -> tuple[list[int], list[int]]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=config.RANDOM_SEED,
    )
    indices = list(range(len(targets)))
    train_indices, val_indices = next(splitter.split(indices, targets))
    return train_indices.tolist(), val_indices.tolist()


def _print_split_distribution(name: str, dataset: Subset) -> None:
    labels = [dataset.dataset.targets[index] for index in dataset.indices]
    counts = Counter(labels)
    readable = {
        dataset.dataset.classes[class_index]: counts[class_index]
        for class_index in sorted(counts)
    }
    print(f"{name} distribution: {readable}")


def _print_dataset_distribution(name: str, dataset: Dataset) -> None:
    counts = Counter(dataset.targets)
    readable = {
        dataset.classes[class_index]: counts[class_index]
        for class_index in sorted(counts)
    }
    print(f"{name} distribution: {readable}")


def _create_manifest_dataloaders(
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dataset = ManifestImageDataset(config.TRAIN_MANIFEST, transform=get_train_transform())
    val_dataset = ManifestImageDataset(config.VAL_MANIFEST, transform=get_eval_transform())
    test_dataset = ManifestImageDataset(config.TEST_MANIFEST, transform=get_eval_transform())

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

    _print_dataset_distribution("Train", train_dataset)
    _print_dataset_distribution("Val", val_dataset)
    _print_dataset_distribution("Test", test_dataset)

    return train_loader, val_loader, test_loader, train_dataset.classes


def _create_imagefolder_dataloaders(
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    if not config.TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {config.TRAIN_DIR}")
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"Testing directory not found: {config.TEST_DIR}")

    train_full = datasets.ImageFolder(config.TRAIN_DIR, transform=get_train_transform())
    val_full = datasets.ImageFolder(config.TRAIN_DIR, transform=get_eval_transform())
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=get_eval_transform())

    _validate_class_names(train_full.classes)
    _validate_class_names(val_full.classes)
    _validate_class_names(test_dataset.classes)

    train_indices, val_indices = _stratified_train_val_indices(train_full.targets)
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

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

    _print_split_distribution("Train", train_dataset)
    _print_split_distribution("Val", val_dataset)
    test_counts = Counter(test_dataset.targets)
    readable_test_counts = {
        test_dataset.classes[class_index]: test_counts[class_index]
        for class_index in sorted(test_counts)
    }
    print(f"Test distribution: {readable_test_counts}")

    return train_loader, val_loader, test_loader, train_full.classes


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    if config.USE_MANIFEST:
        return _create_manifest_dataloaders(batch_size, num_workers)
    return _create_imagefolder_dataloaders(batch_size, num_workers)
