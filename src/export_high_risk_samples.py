from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import config


HIGH_RISK_CSV = config.REPORT_DIR / "high_risk_tumor_to_notumor_manifest.csv"
OUTPUT_DIR = config.OUTPUT_DIR / "high_risk_samples" / "tumor_to_notumor"
SUMMARY_PATH = config.OUTPUT_DIR / "high_risk_samples" / "high_risk_export_summary.txt"


@dataclass(frozen=True)
class HighRiskRow:
    image_path: Path
    true_class: str
    predicted_class: str
    confidence: float


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


def safe_filename(row: HighRiskRow) -> str:
    original_name = row.image_path.name
    return (
        f"{row.true_class}__pred-{row.predicted_class}"
        f"__conf-{row.confidence:.4f}__{original_name}"
    )


def main() -> int:
    rows = read_high_risk_rows(HIGH_RISK_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    failed: list[str] = []

    for row in rows:
        destination = OUTPUT_DIR / safe_filename(row)
        try:
            if not row.image_path.exists():
                raise FileNotFoundError(row.image_path)
            shutil.copy2(row.image_path, destination)
            copied.append(destination)
        except OSError as exc:
            failed.append(f"{row.image_path}: {exc}")

    summary_lines = [
        "High-Risk Sample Export Summary",
        f"Input CSV: {HIGH_RISK_CSV}",
        f"Output folder: {OUTPUT_DIR}",
        f"Rows in CSV: {len(rows)}",
        f"Copied images: {len(copied)}",
        f"Failed images: {len(failed)}",
        "",
        "Raw dataset files were not modified, moved, deleted, or renamed.",
    ]

    if failed:
        summary_lines.extend(["", "Failed samples:", *[f"  {item}" for item in failed]])

    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
