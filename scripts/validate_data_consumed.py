"""Validate the consumed dataset using the derived consumption index.

This validator no longer reads the raw Label Studio export directly. It only
uses the derived `data_consumed/consumption_index.json` plus the image folder.

Checks performed:
1. Image-to-index mapping verification
   - Every image in `data_consumed/images` must have at least one index record.
   - Every index record must point to an image stem that exists on disk.

2. Zero-consumption and missing mask check
   - For every `pct_*` number field, a value of 0 must be backed by a matching
     polygon label for the same item.

3. Mutually exclusive state validation
   - A field such as `drink_water`, `status_cookie`, or `extra_butter` may not
     contain both `Consumed` and `Not consumed` across the same image records.

The script writes a log and a JSON summary to `data_consumed/audit_logs/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_DATASET_DIR = Path("data_consumed")
DEFAULT_IMAGES_DIR = DEFAULT_DATASET_DIR / "images"
DEFAULT_INDEX_PATH = DEFAULT_DATASET_DIR / "consumption_index.json"
DEFAULT_LOG_DIR = DEFAULT_DATASET_DIR / "audit_logs"


@dataclass
class AuditIssue:
    check_name: str
    severity: str
    item: str
    message: str


@dataclass
class AuditResult:
    image_files_checked: int
    indexed_images_checked: int
    issues: List[AuditIssue]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate data_consumed using the derived consumption index."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the derived consumption index JSON.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory where audit logs are written.",
    )
    return parser.parse_args()


def get_image_stem_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def flatten_choice_values(choice_values: Sequence[Sequence[str]]) -> List[str]:
    flattened: List[str] = []
    for item in choice_values:
        flattened.extend(item)
    return flattened


def load_index(index_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Consumption index not found: {index_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = payload.get("indexed_by_image_stem", {})
    if not isinstance(indexed, dict):
        raise ValueError("consumption_index.json does not contain indexed_by_image_stem")

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for stem, records in indexed.items():
        if isinstance(records, list):
            normalized[stem] = [record for record in records if isinstance(record, dict)]
    return normalized


def get_image_paths_by_stem(images_dir: Path) -> Dict[str, Path]:
    image_paths_by_stem: Dict[str, Path] = {}
    for image_path in images_dir.iterdir():
        if image_path.is_file():
            image_paths_by_stem[get_image_stem_from_name(image_path.name)] = image_path
    return image_paths_by_stem


def audit_indexed_images(
    image_paths_by_stem: Dict[str, Path],
    indexed_records: Dict[str, List[Dict[str, Any]]],
) -> List[AuditIssue]:
    issues: List[AuditIssue] = []

    for image_stem, image_path in image_paths_by_stem.items():
        if image_stem not in indexed_records:
            issues.append(
                AuditIssue(
                    check_name="image_to_index_mapping",
                    severity="error",
                    item=image_path.name,
                    message="No consumption record was found for this image stem.",
                )
            )

    for image_stem, records in indexed_records.items():
        if image_stem not in image_paths_by_stem:
            issues.append(
                AuditIssue(
                    check_name="image_to_index_mapping",
                    severity="error",
                    item=image_stem,
                    message="The index contains a record for an image that does not exist on disk.",
                )
            )
            continue

        image_name = image_paths_by_stem[image_stem].name

        for record in records:
            polygon_labels = record.get("polygon_labels", [])
            numbers = record.get("numbers", {})
            choices = record.get("choices", {})

            if not isinstance(polygon_labels, list):
                polygon_labels = []
            if not isinstance(numbers, dict):
                numbers = {}
            if not isinstance(choices, dict):
                choices = {}

            # Check 2: zero percent values need a matching polygon label.
            for metric_name, metric_values in numbers.items():
                item_name = metric_name[4:] if metric_name.startswith("pct_") else metric_name
                for metric_value in metric_values:
                    if isinstance(metric_value, (int, float)) and metric_value == 0 and item_name not in polygon_labels:
                        issues.append(
                            AuditIssue(
                                check_name="zero_consumption_and_missing_mask",
                                severity="error",
                                item=image_name,
                                message=(
                                    f"{metric_name} is 0 but the polygon label '{item_name}' is "
                                    "missing from polygon_labels."
                                ),
                            )
                        )

            # Check 3: a single field may not contain both consumed and not consumed.
            for field_name, choice_values in choices.items():
                if not isinstance(choice_values, list):
                    continue

                flattened = flatten_choice_values(choice_values)
                normalized = {str(choice).strip() for choice in flattened}
                if "Consumed" in normalized and "Not consumed" in normalized:
                    issues.append(
                        AuditIssue(
                            check_name="mutually_exclusive_state_validation",
                            severity="error",
                            item=f"{image_name}:{field_name}",
                            message=(
                                "The same choice field contains both 'Consumed' and 'Not consumed'."
                            ),
                        )
                    )

    return issues


def write_audit_log(log_dir: Path, result: AuditResult) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"data_consumed_audit_{timestamp}.log"
    json_path = log_dir / f"data_consumed_audit_{timestamp}.json"

    lines: List[str] = []
    lines.append("DATA CONSUMED AUDIT REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Image files checked: {result.image_files_checked}")
    lines.append(f"Indexed images checked: {result.indexed_images_checked}")
    lines.append(f"Issues found: {len(result.issues)}")
    lines.append("")

    if result.issues:
        lines.append("Issues:")
        for index, issue in enumerate(result.issues, start=1):
            lines.append(f"{index}. [{issue.severity.upper()}] {issue.check_name} | {issue.item}")
            lines.append(f"   {issue.message}")
    else:
        lines.append("No anomalies detected.")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_files_checked": result.image_files_checked,
        "indexed_images_checked": result.indexed_images_checked,
        "issues": [asdict(issue) for issue in result.issues],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return log_path, json_path


def print_summary(result: AuditResult, log_path: Path, json_path: Path) -> None:
    print("Data consumed audit complete.")
    print(f"  Images checked   : {result.image_files_checked}")
    print(f"  Indexed checked  : {result.indexed_images_checked}")
    print(f"  Issues found     : {len(result.issues)}")
    print(f"  Log file         : {log_path}")
    print(f"  JSON summary     : {json_path}")

    if result.issues:
        print("\nFirst issues:")
        for issue in result.issues[:10]:
            print(f"  - [{issue.severity.upper()}] {issue.check_name} -> {issue.item}")


def main() -> int:
    args = parse_args()

    dataset_dir = args.dataset_dir.resolve()
    images_dir = args.images_dir.resolve()
    index_path = args.index_path.resolve()
    log_dir = args.log_dir.resolve()

    if not dataset_dir.exists():
        print(f"Dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 2
    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}", file=sys.stderr)
        return 2

    indexed_records = load_index(index_path)
    image_paths_by_stem = get_image_paths_by_stem(images_dir)

    issues = audit_indexed_images(image_paths_by_stem, indexed_records)
    result = AuditResult(
        image_files_checked=len(image_paths_by_stem),
        indexed_images_checked=len(indexed_records),
        issues=issues,
    )

    log_path, json_path = write_audit_log(log_dir, result)
    print_summary(result, log_path, json_path)

    return 1 if result.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())