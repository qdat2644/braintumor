from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit(
        "Pillow is required to run this audit. Install it with: pip install pillow"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "extracted"
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "dataset_audit.txt"

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


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    class_name: str
    md5: str | None
    size: tuple[int, int] | None
    error: str | None


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)
        print(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def class_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(
        (path for path in split_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )


def image_files_for_class(class_dir: Path) -> list[Path]:
    return sorted(
        (path for path in class_dir.rglob("*") if is_image_file(path)),
        key=lambda path: str(path).lower(),
    )


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def audit_split(split_name: str, split_dir: Path) -> tuple[Counter[str], list[ImageRecord]]:
    distribution: Counter[str] = Counter()
    records: list[ImageRecord] = []

    for class_dir in class_dirs(split_dir):
        class_name = class_dir.name
        files = image_files_for_class(class_dir)
        distribution[class_name] = len(files)

        for image_path in files:
            md5_value: str | None = None
            size: tuple[int, int] | None = None
            error: str | None = None

            try:
                md5_value = compute_md5(image_path)
            except OSError as exc:
                error = f"Could not read file bytes: {exc}"

            try:
                size = read_image_size(image_path)
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                if error:
                    error = f"{error}; PIL open failed: {exc}"
                else:
                    error = f"PIL open failed: {exc}"

            records.append(
                ImageRecord(
                    path=image_path,
                    split=split_name,
                    class_name=class_name,
                    md5=md5_value,
                    size=size,
                    error=error,
                )
            )

    return distribution, records


def find_duplicates(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.md5 is not None:
            groups[record.md5].append(record)
    return {md5: group for md5, group in groups.items() if len(group) > 1}


def find_cross_split_duplicates(
    train_records: list[ImageRecord], test_records: list[ImageRecord]
) -> dict[str, list[ImageRecord]]:
    train_hashes = {record.md5 for record in train_records if record.md5 is not None}
    test_hashes = {record.md5 for record in test_records if record.md5 is not None}
    shared_hashes = train_hashes & test_hashes

    records_by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in train_records + test_records:
        if record.md5 in shared_hashes:
            records_by_hash[record.md5].append(record)

    return dict(sorted(records_by_hash.items()))


def add_distribution(report: Report, split_name: str, distribution: Counter[str]) -> None:
    report.add(f"{split_name} class distribution:")
    if not distribution:
        report.add("  No class folders found.")
        return

    total = sum(distribution.values())
    for class_name, count in sorted(distribution.items(), key=lambda item: item[0].lower()):
        report.add(f"  {class_name}: {count}")
    report.add(f"  Total: {total}")


def add_corruption_report(report: Report, records: list[ImageRecord]) -> None:
    corrupted = [record for record in records if record.error is not None]
    report.add(f"Corrupted/unreadable images: {len(corrupted)}")
    for record in corrupted:
        report.add(f"  {relative_path(record.path)}")
        report.add(f"    {record.error}")


def add_duplicate_report(
    report: Report, title: str, duplicate_groups: dict[str, list[ImageRecord]]
) -> None:
    report.add(title)
    if not duplicate_groups:
        report.add("  No duplicates found.")
        return

    report.add(f"  Duplicate hash groups: {len(duplicate_groups)}")
    for md5_value, records in sorted(duplicate_groups.items()):
        report.add(f"  md5={md5_value}")
        for record in sorted(records, key=lambda item: str(item.path).lower()):
            report.add(f"    [{record.split}/{record.class_name}] {relative_path(record.path)}")


def add_size_stats(report: Report, title: str, records: list[ImageRecord]) -> None:
    sizes = [record.size for record in records if record.size is not None]
    report.add(title)
    if not sizes:
        report.add("  No readable image sizes found.")
        return

    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    common_sizes = Counter(sizes).most_common(10)

    report.add(f"  Readable images: {len(sizes)}")
    report.add(f"  Minimum width/height: {min(widths)}x{min(heights)}")
    report.add(f"  Maximum width/height: {max(widths)}x{max(heights)}")
    report.add("  Most common image sizes:")
    for (width, height), count in common_sizes:
        report.add(f"    {width}x{height}: {count}")


def add_folder_check(report: Report) -> bool:
    report.add("Expected dataset folders:")
    all_exist = True
    for path in (TRAIN_DIR, TEST_DIR):
        exists = path.exists() and path.is_dir()
        status = "OK" if exists else "MISSING"
        report.add(f"  [{status}] {relative_path(path)}")
        all_exist = all_exist and exists

    if not all_exist:
        root_train = PROJECT_ROOT / "Training"
        root_test = PROJECT_ROOT / "Testing"
        if root_train.exists() or root_test.exists():
            report.add()
            report.add("Note: Training/Testing folders were found at the project root.")
            report.add("This script intentionally checks only data/extracted/Training and data/extracted/Testing.")

    return all_exist


def main() -> int:
    report = Report()
    report.add("Dataset Audit Report")
    report.add(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    report.add(f"Project root: {PROJECT_ROOT}")
    report.add()

    folders_exist = add_folder_check(report)
    report.add()

    if not folders_exist:
        report.add("Audit stopped because one or more expected dataset folders are missing.")
        report.save(REPORT_PATH)
        report.add()
        report.add(f"Report saved to: {relative_path(REPORT_PATH)}")
        return 1

    train_distribution, train_records = audit_split("Training", TRAIN_DIR)
    test_distribution, test_records = audit_split("Testing", TEST_DIR)
    all_records = train_records + test_records

    add_distribution(report, "Training", train_distribution)
    report.add()
    add_distribution(report, "Testing", test_distribution)
    report.add()

    add_corruption_report(report, all_records)
    report.add()

    add_duplicate_report(report, "Duplicates inside Training:", find_duplicates(train_records))
    report.add()
    add_duplicate_report(report, "Duplicates inside Testing:", find_duplicates(test_records))
    report.add()
    add_duplicate_report(
        report,
        "Duplicates across Training and Testing:",
        find_cross_split_duplicates(train_records, test_records),
    )
    report.add()

    add_size_stats(report, "Training image size statistics:", train_records)
    report.add()
    add_size_stats(report, "Testing image size statistics:", test_records)
    report.add()
    add_size_stats(report, "Overall image size statistics:", all_records)
    report.add()

    report.save(REPORT_PATH)
    report.add(f"Report saved to: {relative_path(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
