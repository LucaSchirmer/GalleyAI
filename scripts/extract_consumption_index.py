"""Convert the Label Studio export into an image-keyed consumption index.

The original JSON export remains the source of truth. This script creates a
derived JSON file that is easier to use for validation and lookup:

    data_consumed/consumption_index.json

The derived file is keyed by the original image filename stem, so you can look
up a record by the image you already have.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_PATH = SCRIPT_DIR.parent / "project-1-at-2026-08-28-10-35-6a60389c.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parent / "data_consumed" / "consumption_index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-keyed consumption index from the Label Studio export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help="Path to the original Label Studio JSON export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the derived mapping JSON should be written.",
    )
    return parser.parse_args()


def extract_source_file(task: Dict[str, Any]) -> str:
    """Return the consumed image filename stored in the task's data block.

    The export usually contains both `consumed` and `unconsumed` image URLs.
    For the lookup index we prefer the consumed side, because that is the image
    you want to map validation data against.
    """

    data = task.get("data", {})
    preferred_keys = ("consumed", "image", "img", "img_consumed", "data")

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, str) and value.startswith("http"):
            parsed_url = urlparse(value)
            return Path(unquote(parsed_url.path)).name

    for value in data.values():
        if isinstance(value, str) and value.startswith("http"):
            parsed_url = urlparse(value)
            return Path(unquote(parsed_url.path)).name

    return ""


def build_record(task: Dict[str, Any]) -> Dict[str, Any]:
    """Collect the consumption-related fields for one Label Studio task."""

    source_file = extract_source_file(task)
    source_stem = Path(source_file).stem if source_file else ""

    polygon_labels: List[str] = []
    numbers: Dict[str, List[float]] = defaultdict(list)
    choices: Dict[str, List[List[str]]] = defaultdict(list)
    raw_results: List[Dict[str, Any]] = []

    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            result_type = result.get("type")
            value = result.get("value", {})
            from_name = result.get("from_name", "")

            raw_results.append(result)

            if result_type == "polygonlabels":
                polygon_labels.extend(value.get("polygonlabels", []))
            elif result_type == "number":
                number_value = value.get("number")
                if number_value is not None:
                    numbers[from_name].append(number_value)
            elif result_type == "choices":
                choice_value = value.get("choices", [])
                if choice_value:
                    choices[from_name].append(choice_value)

    return {
        "task_id": task.get("id"),
        "source_file": source_file,
        "source_stem": source_stem,
        "polygon_labels": polygon_labels,
        "numbers": dict(numbers),
        "choices": dict(choices),
        "annotation_count": len(task.get("annotations", [])),
        "raw_result_count": len(raw_results),
    }


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input export not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as file_handle:
        tasks = json.load(file_handle)

    indexed_tasks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unnamed_tasks: List[Dict[str, Any]] = []

    for task in tasks:
        record = build_record(task)

        if record["source_stem"]:
            indexed_tasks[record["source_stem"]].append(record)
        else:
            unnamed_tasks.append(record)

    output_payload = {
        "source_export": str(args.input),
        "indexed_by_image_stem": indexed_tasks,
        "unnamed_tasks": unnamed_tasks,
        "task_count": len(tasks),
        "indexed_image_count": len(indexed_tasks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_handle:
        json.dump(output_payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")

    print(f"Wrote consumption index to: {args.output}")
    print(f"Tasks processed: {len(tasks)}")
    print(f"Images indexed : {len(indexed_tasks)}")
    if unnamed_tasks:
        print(f"Unnamed tasks  : {len(unnamed_tasks)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())