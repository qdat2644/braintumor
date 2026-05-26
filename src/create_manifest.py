from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

import config


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

CSV_COLUMNS = ["path", "split", "class_name", "class_idx", "md5", "width", "height"]
VAL_FRACTION = 0.15


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    class_name: str
    class_idx: int
    md5: str
    width: int
    height: int

    def to_csv_row(self, split: str | None = None) -> dict[str, object]:
        return {
            "path": self.path.relative_to(config.PROJECT_ROOT).as_posix(),
            "split": self.split if split is None else split,
            "class_name": self.class_name,
            "class_idx": self.class_idx,
            "md5": self.md5,
            "width": self.width,
            "height": self.height,
        }


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def collect_records(split: str, split_dir: Path) -> list[ImageRecord]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_dir}")

    records: list[ImageRecord] = []
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        image_paths = sorted(
            (path for path in class_dir.rglob("*") if is_image_file(path)),
            key=lambda path: path.relative_to(config.PROJECT_ROOT).as_posix().lower(),
        )
        for image_path in image_paths:
            width, height = read_image_size(image_path)
            records.append(
                ImageRecord(
                    path=image_path,
                    split=split,
                    class_name=class_name,
                    class_idx=class_idx,
                    md5=compute_md5(image_path),
                    width=width,
                    height=height,
                )
            )

    return records


def deduplicate_records(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], Counter[str], dict[str, list[ImageRecord]]]:
    kept: list[ImageRecord] = []
    removed_by_class: Counter[str] = Counter()
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    seen_hashes: set[str] = set()

    for record in records:
        groups[record.md5].append(record)
        if record.md5 in seen_hashes:
            removed_by_class[record.class_name] += 1
            continue
        seen_hashes.add(record.md5)
        kept.append(record)

    duplicate_groups = {
        md5_value: group for md5_value, group in groups.items() if len(group) > 1
    }
    return kept, removed_by_class, duplicate_groups


def stratified_train_val_split(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(config.RANDOM_SEED)
    by_class: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_idx].append(record)

    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    for class_idx in range(len(config.CLASS_NAMES)):
        class_records = list(by_class[class_idx])
        rng.shuffle(class_records)
        val_count = round(len(class_records) * VAL_FRACTION)
        val_records.extend(class_records[:val_count])
        train_records.extend(class_records[val_count:])

    train_records.sort(key=lambda record: record.path.as_posix().lower())
    val_records.sort(key=lambda record: record.path.as_posix().lower())
    return train_records, val_records


def count_by_class(records: list[ImageRecord]) -> Counter[str]:
    return Counter(record.class_name for record in records)


def write_manifest(path: Path, records: list[ImageRecord], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row(split=split))


def format_counts(counts: Counter[str]) -> list[str]:
    return [f"  {class_name}: {counts[class_name]}" for class_name in config.CLASS_NAMES]


def write_report(
    original_train: list[ImageRecord],
    clean_train: list[ImageRecord],
    removed_train: Counter[str],
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    train_duplicate_groups: dict[str, list[ImageRecord]],
    original_test: list[ImageRecord],
    clean_test: list[ImageRecord],
    removed_test: Counter[str],
    test_duplicate_groups: dict[str, list[ImageRecord]],
) -> None:
    report_path = config.REPORT_DIR / "manifest_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "Manifest Cleaning Report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Project root: {config.PROJECT_ROOT}",
        "",
        "Raw file policy:",
        "  Raw dataset files were not modified, moved, deleted, or renamed.",
        "  Duplicates were removed only from CSV manifests.",
        "",
        "Original Training count per class:",
        *format_counts(count_by_class(original_train)),
        "",
        "Cleaned Training count per class:",
        *format_counts(count_by_class(clean_train)),
        "",
        "Removed Training duplicates per class:",
        *format_counts(removed_train),
        f"  Duplicate hash groups inside Training: {len(train_duplicate_groups)}",
        "",
        "Original Testing count per class:",
        *format_counts(count_by_class(original_test)),
        "",
        "Cleaned Testing count per class:",
        *format_counts(count_by_class(clean_test)),
        "",
        "Removed Testing duplicates per class:",
        *format_counts(removed_test),
        f"  Duplicate hash groups inside Testing: {len(test_duplicate_groups)}",
        "",
        "Final manifest counts:",
        f"  train_manifest.csv: {len(train_records)}",
        f"  val_manifest.csv: {len(val_records)}",
        f"  test_manifest.csv: {len(clean_test)}",
        "",
        "Manifest paths:",
        f"  {config.TRAIN_MANIFEST}",
        f"  {config.VAL_MANIFEST}",
        f"  {config.TEST_MANIFEST}",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    original_train = collect_records("Training", config.TRAIN_DIR)
    original_test = collect_records("Testing", config.TEST_DIR)

    clean_train, removed_train, train_duplicate_groups = deduplicate_records(
        original_train
    )
    clean_test, removed_test, test_duplicate_groups = deduplicate_records(original_test)

    train_records, val_records = stratified_train_val_split(clean_train)

    write_manifest(config.TRAIN_MANIFEST, train_records, split="train")
    write_manifest(config.VAL_MANIFEST, val_records, split="val")
    write_manifest(config.TEST_MANIFEST, clean_test, split="test")

    write_report(
        original_train=original_train,
        clean_train=clean_train,
        removed_train=removed_train,
        train_records=train_records,
        val_records=val_records,
        train_duplicate_groups=train_duplicate_groups,
        original_test=original_test,
        clean_test=clean_test,
        removed_test=removed_test,
        test_duplicate_groups=test_duplicate_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
