from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import config


BINARY_COLUMNS = [
    "path",
    "split",
    "original_class_name",
    "binary_class_name",
    "binary_class_idx",
    "md5",
    "width",
    "height",
]


def binary_label(class_name: str) -> tuple[str, int]:
    if class_name == "notumor":
        return "notumor", 1
    if class_name in {"glioma", "meningioma", "pituitary"}:
        return "tumor", 0
    raise ValueError(f"Unknown original class name: {class_name}")


def convert_manifest(input_path: Path, output_path: Path) -> Counter[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    with input_path.open("r", newline="", encoding="utf-8") as src_file:
        reader = csv.DictReader(src_file)
        required_columns = {"path", "split", "class_name", "md5", "width", "height"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{input_path} is missing required columns: {sorted(missing_columns)}"
            )

        with output_path.open("w", newline="", encoding="utf-8") as dst_file:
            writer = csv.DictWriter(dst_file, fieldnames=BINARY_COLUMNS)
            writer.writeheader()

            for row in reader:
                binary_class_name, binary_class_idx = binary_label(row["class_name"])
                counts[binary_class_name] += 1
                writer.writerow(
                    {
                        "path": row["path"],
                        "split": row["split"],
                        "original_class_name": row["class_name"],
                        "binary_class_name": binary_class_name,
                        "binary_class_idx": binary_class_idx,
                        "md5": row["md5"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )

    return counts


def main() -> int:
    conversions = [
        ("train", config.TRAIN_MANIFEST, config.BINARY_TRAIN_MANIFEST),
        ("val", config.VAL_MANIFEST, config.BINARY_VAL_MANIFEST),
        ("test", config.TEST_MANIFEST, config.BINARY_TEST_MANIFEST),
    ]

    print("Creating binary tumor-vs-notumor manifests.")
    for split_name, input_path, output_path in conversions:
        counts = convert_manifest(input_path, output_path)
        print(f"{split_name}: {output_path}")
        for class_name in config.BINARY_CLASS_NAMES:
            print(f"  {class_name}: {counts[class_name]}")

    print("Raw dataset files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
