"""Web UI to validate and edit consumed-image annotations.

This tool reads the pre-built consumption index JSON
(`data_consumed/consumption_index.json`, shape: {"source_export": ...,
"indexed_by_image_stem": {stem: [record]}}) and provides:

1. Validation of mask-vs-consumption consistency per stem.
2. Editing of numeric consumption fields (pct_*) and choice fields.
3. Save-back to the same JSON file (in place by default).

The UI joins the consumption index with the generated pair manifests and
cached YOLO detections when those files are available. Detector-derived
findings are warnings; deterministic annotation problems are errors.

Run:
    python scripts/review_consumption_webui.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import quote, unquote


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data_consumed" / "consumption_index.json"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data_consumed" / "images"
DEFAULT_BEFORE_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data_pairs_with_splits"
DEFAULT_MASK_CACHE_DIR = PROJECT_ROOT / "mask_cache"


# Metric aliases used by validation when metric field name differs from polygon label names.
# Label Studio and YOLO use different names for a few classes. Keep the two
# vocabularies separate: polygon_labels come from Label Studio, while mask
# cache detections use the YOLO names.
METRIC_ANNOTATION_LABELS: Dict[str, List[str]] = {
    "pct_vanilla_pudding": ["vanilla_pudding_with_fruits"],
    "pct_salad_dish_main": ["salad_main"],
    # Composite dish metric: any component mask is acceptable.
    "pct_chicken_rice_veg": ["chicken", "rice", "carrots", "broccoli"],
    "pct_brownie": ["brownie"],
}

METRIC_MODEL_CLASSES: Dict[str, List[str]] = {
    "pct_vanilla_pudding": ["vanilla_pudding_with_fruits"],
    "pct_salad_dish_main": ["main_salad"],
    "pct_chicken_rice_veg": ["chicken", "rice", "carrots", "broccoli"],
    "pct_brownie": ["chocolate_cake"],
}

# Default vocabulary for simple consumption-state choice fields. Any field
# that already uses one of these values gets the rest offered too, so the
# reviewer can switch between them. Fields with a different vocabulary
# (e.g. quality_flags) just use whatever values were observed in the data.
CHOICE_DEFAULT_OPTIONS = ["Consumed", "Not consumed", "Not present"]
CHOICE_FIELD_TO_CLASS: Dict[str, str] = {
    "drink_water": "water",
    "drink_coffee": "coffee",
    "drink_tea": "tea",
    "drink_oj": "orange_juice",
    "drink_cola": "cola",
    "extra_butter": "butter",
    "extra_honey": "honey",
    "extra_plum_jam": "plum_jam",
    "extra_cherry_jam": "cherry_jam",
    "status_cookie": "cookie",
}

CLEAN_QUALITY_FLAG = "No issues — image pair is clean"
QUALITY_FLAG_OPTIONS = [
    CLEAN_QUALITY_FLAG,
    "Blurry",
    "Food rearranged significantly",
    "Lighting inconsistency vs other images",
    "Non-edible residuals present (bones, core, crumbs only)",
    "Obstacle obscuring food (cutlery, napkin, foil)",
    "Odd tray angle",
    "Shadow on tray",
    "Tray partially out of frame",
]

# Tray configuration vocabulary -> required annotation field / YOLO class.
POSSIBLE_ELEMENT_FIELDS: Dict[str, str] = {
    "Bread roll": "pct_bread_roll",
    "Fruit salad": "pct_fruit_salad",
    "Chocolate cake": "pct_brownie",
    "Vanilla pudding with fruits": "pct_vanilla_pudding",
    "Vanilla puddding with fruits": "pct_vanilla_pudding",  # source typo
    "Side salad": "pct_side_salad",
    "Salad": "pct_salad_dish_main",
    "Cookie": "status_cookie",
    "Water": "drink_water",
    "Coffee": "drink_coffee",
    "Tea": "drink_tea",
    "Orange juice": "drink_oj",
    "Cola": "drink_cola",
    "Butter": "extra_butter",
    "Honey": "extra_honey",
    "Plum jam": "extra_plum_jam",
    "Cherry jam": "extra_cherry_jam",
}

POSSIBLE_ELEMENT_CLASSES: Dict[str, str] = {
    "Bread roll": "bread_roll",
    "Fruit salad": "fruit_salad",
    "Chocolate cake": "chocolate_cake",
    "Vanilla pudding with fruits": "vanilla_pudding_with_fruits",
    "Vanilla puddding with fruits": "vanilla_pudding_with_fruits",
    "Side salad": "side_salad",
    "Salad": "main_salad",
    "Cookie": "cookie",
    "Water": "water",
    "Coffee": "coffee",
    "Tea": "tea",
    "Orange juice": "orange_juice",
    "Cola": "cola",
    "Butter": "butter",
    "Honey": "honey",
    "Plum jam": "plum_jam",
    "Cherry jam": "cherry_jam",
    "Chicken": "chicken",
    "Rice": "rice",
    "Broccoli": "broccoli",
    "Carrots": "carrots",
}

MAIN_DISH_ELEMENTS = {"Chicken", "Rice", "Broccoli", "Carrots"}
KNOWN_CONSUMPTION_FIELDS = set(POSSIBLE_ELEMENT_FIELDS.values()) | {
    "pct_chicken_rice_veg"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review consumed-image annotations with a local web UI."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--before-images-dir", type=Path, default=DEFAULT_BEFORE_IMAGES_DIR
    )
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to overwriting --input in place.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def suffix_stem(name: str) -> str:
    stem = Path(name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


class ReviewSession:
    def __init__(
        self,
        input_path: Path,
        images_dir: Path,
        output_path: Path | None,
        before_images_dir: Path = DEFAULT_BEFORE_IMAGES_DIR,
        pairs_dir: Path = DEFAULT_PAIRS_DIR,
        mask_cache_dir: Path = DEFAULT_MASK_CACHE_DIR,
    ):
        self.input_path = input_path
        self.images_dir = images_dir
        self.before_images_dir = before_images_dir
        self.pairs_dir = pairs_dir
        self.mask_cache_dir = mask_cache_dir
        self.output_path = output_path or input_path

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input index not found: {self.input_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        with self.input_path.open("r", encoding="utf-8") as fh:
            self.data: Dict[str, Any] = json.load(fh)

        raw_indexed = self.data.get("indexed_by_image_stem", {})
        if not isinstance(raw_indexed, dict):
            raise ValueError("indexed_by_image_stem must be a JSON object")

        self.indexed: Dict[str, Dict[str, Any]] = {}
        self.extra_records_by_stem: Dict[str, List[Dict[str, Any]]] = {}
        self.duplicate_record_counts: Dict[str, int] = {}
        for stem, records in raw_indexed.items():
            valid_records = [record for record in records if isinstance(record, dict)]
            if not valid_records:
                continue
            self.indexed[stem] = valid_records[0]
            self.extra_records_by_stem[stem] = valid_records[1:]
            if len(valid_records) > 1:
                self.duplicate_record_counts[stem] = len(valid_records)

        self.stems = sorted(self.indexed.keys())
        self.after_image_paths = self._index_images(self.images_dir)
        self.before_image_paths = self._index_images(self.before_images_dir)
        self.pairs_by_after = self._load_pairs()
        self.mask_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.choice_options = self._collect_choice_options()
        self.dataset_warnings = self._dataset_warnings()
        self._summary_cache: List[Dict[str, Any]] | None = None

    @staticmethod
    def _index_images(images_dir: Path) -> Dict[str, Path]:
        indexed: Dict[str, Path] = {}
        if not images_dir.exists():
            return indexed
        for path in images_dir.iterdir():
            if path.is_file():
                indexed[suffix_stem(path.name)] = path
        return indexed

    @staticmethod
    def _resolve_project_path(raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _load_pairs(self) -> Dict[str, List[Dict[str, Any]]]:
        pairs_by_after: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if not self.pairs_dir.exists():
            return pairs_by_after

        for manifest_path in sorted(self.pairs_dir.glob("*.json")):
            try:
                pairs = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, dict) or not pair.get("after"):
                    continue
                after_stem = suffix_stem(str(pair["after"]))
                pair_info = dict(pair)
                pair_info["split"] = manifest_path.stem
                pair_info["before_path"] = self._resolve_project_path(pair.get("before"))
                pair_info["after_path"] = self._resolve_project_path(pair.get("after"))
                pairs_by_after[after_stem].append(pair_info)
                before_path = pair_info["before_path"]
                if before_path is not None and before_path.exists():
                    self.before_image_paths[suffix_stem(before_path.name)] = before_path
        return pairs_by_after

    def _dataset_warnings(self) -> List[str]:
        warnings: List[str] = []
        declared = self.data.get("task_count")
        record_count = sum(
            1 + len(self.extra_records_by_stem.get(stem, [])) for stem in self.indexed
        )
        if isinstance(declared, int) and declared != record_count:
            warnings.append(
                f"Index declares {declared} source tasks but contains {record_count} records. "
                "Some duplicate-stem tasks may have been collapsed."
            )
        if not self.pairs_dir.exists():
            warnings.append("Pair manifests were not found; before-image checks are disabled.")
        if not self.mask_cache_dir.exists():
            warnings.append("Mask cache was not found; detector-based checks are disabled.")
        return warnings

    def _collect_choice_options(self) -> Dict[str, List[str]]:
        options: Dict[str, List[str]] = {}
        for record in self.indexed.values():
            for field, values in record.get("choices", {}).items():
                selected = values[0] if values else []
                opts = options.setdefault(field, [])
                for value in selected:
                    if value not in opts:
                        opts.append(value)

        for field, opts in options.items():
            if field in CHOICE_FIELD_TO_CLASS:
                for default in CHOICE_DEFAULT_OPTIONS:
                    if default not in opts:
                        opts.append(default)
        quality_options = options.setdefault("quality_flags", [])
        for quality_flag in QUALITY_FLAG_OPTIONS:
            if quality_flag not in quality_options:
                quality_options.append(quality_flag)
        return options

    @staticmethod
    def _annotation_labels(metric_name: str) -> List[str]:
        if metric_name in METRIC_ANNOTATION_LABELS:
            return METRIC_ANNOTATION_LABELS[metric_name]
        return [metric_name[4:] if metric_name.startswith("pct_") else metric_name]

    @staticmethod
    def _model_classes(metric_name: str) -> List[str]:
        if metric_name in METRIC_MODEL_CLASSES:
            return METRIC_MODEL_CLASSES[metric_name]
        return [metric_name[4:] if metric_name.startswith("pct_") else metric_name]

    @staticmethod
    def _selected_values(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(value) for group in values if isinstance(group, list) for value in group]

    def _detections_for_stem(self, stem: str) -> List[Dict[str, Any]]:
        if stem not in self.mask_cache:
            cache_path = self.mask_cache_dir / f"{stem}.json"
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                detections = payload.get("detections", [])
                self.mask_cache[stem] = [d for d in detections if isinstance(d, dict)]
            except (OSError, json.JSONDecodeError):
                self.mask_cache[stem] = []
        return self.mask_cache[stem]

    def _classes_for_stem(self, stem: str) -> Set[str]:
        return {
            str(detection["class"])
            for detection in self._detections_for_stem(stem)
            if detection.get("class")
        }

    @staticmethod
    def _issue(
        issues: List[Dict[str, str]], severity: str, check: str, message: str
    ) -> None:
        issues.append({"severity": severity, "check": check, "message": message})

    def _pair_context(self, stem: str) -> Dict[str, Any]:
        pair_infos = self.pairs_by_after.get(stem, [])
        possible_elements = sorted(
            {
                str(element)
                for pair in pair_infos
                for element in pair.get("possible_elements", [])
            }
        )
        before_paths = {
            str(pair["before_path"])
            for pair in pair_infos
            if pair.get("before_path") is not None
        }
        primary_before = next(
            (pair.get("before_path") for pair in pair_infos if pair.get("before_path")),
            None,
        )
        return {
            "pairs": pair_infos,
            "possible_elements": possible_elements,
            "before_paths": before_paths,
            "primary_before": primary_before,
            "splits": sorted({str(pair.get("split")) for pair in pair_infos}),
            "categories": sorted({str(pair.get("category")) for pair in pair_infos}),
        }

    def validate_record(self, stem: str, record: Dict[str, Any]) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        numbers = record.get("numbers", {})
        choices = record.get("choices", {})
        polygon_labels = record.get("polygon_labels", [])
        if not isinstance(numbers, dict):
            numbers = {}
            self._issue(issues, "error", "invalid_numbers", "numbers must be an object.")
        if not isinstance(choices, dict):
            choices = {}
            self._issue(issues, "error", "invalid_choices", "choices must be an object.")
        polygon_label_set = set(polygon_labels) if isinstance(polygon_labels, list) else set()

        if stem in self.duplicate_record_counts:
            self._issue(
                issues,
                "error",
                "duplicate_index_records",
                f"This image stem has {self.duplicate_record_counts[stem]} index records; "
                "resolve them instead of silently choosing one.",
            )

        valid_metric_values: Dict[str, float] = {}
        for metric_name, values in numbers.items():
            if not isinstance(values, list) or len(values) != 1:
                self._issue(
                    issues,
                    "error",
                    "numeric_cardinality",
                    f"{metric_name} must contain exactly one numeric value.",
                )
                continue
            metric_value = values[0]
            if (
                not isinstance(metric_value, (int, float))
                or isinstance(metric_value, bool)
                or not math.isfinite(metric_value)
                or not 0 <= metric_value <= 100
            ):
                self._issue(
                    issues,
                    "error",
                    "invalid_percentage",
                    f"{metric_name}={metric_value!r}; percentages must be finite and within 0–100.",
                )
                continue
            valid_metric_values[metric_name] = float(metric_value)
            expected = self._annotation_labels(metric_name)
            has_expected_mask = bool(polygon_label_set.intersection(expected))
            if metric_value == 0 and not has_expected_mask:
                self._issue(
                    issues,
                    "error",
                    "zero_consumption_and_missing_mask",
                    f"{metric_name}=0 but no expected annotation mask {expected} was found.",
                )
            elif metric_value < 100 and not has_expected_mask:
                self._issue(
                    issues,
                    "warning",
                    "partial_consumption_and_missing_mask",
                    f"{metric_name}={metric_value:g}, so some food should remain, but no "
                    f"expected annotation mask {expected} was found.",
                )

        quality_selected = self._selected_values(choices.get("quality_flags", []))
        if "quality_flags" not in choices or not quality_selected:
            self._issue(
                issues,
                "warning",
                "missing_quality_assessment",
                "No quality assessment is recorded for this image pair.",
            )
        else:
            problem_flags = [flag for flag in quality_selected if flag != CLEAN_QUALITY_FLAG]
            if CLEAN_QUALITY_FLAG in quality_selected and problem_flags:
                self._issue(
                    issues,
                    "error",
                    "conflicting_quality_flags",
                    "The clean-image flag cannot be selected together with a quality problem.",
                )
            for flag in problem_flags:
                self._issue(issues, "quality", "quality_flag", flag)

        for field_name, values in choices.items():
            if field_name == "quality_flags":
                continue
            selected = self._selected_values(values)
            if field_name in CHOICE_FIELD_TO_CLASS:
                states = [value for value in selected if value in CHOICE_DEFAULT_OPTIONS]
                if len(states) != 1 or len(selected) != 1:
                    self._issue(
                        issues,
                        "error",
                        "consumption_choice_cardinality",
                        f"Field '{field_name}' must contain exactly one consumption state.",
                    )

        pair_context = self._pair_context(stem)
        pair_infos = pair_context["pairs"]
        possible_elements = set(pair_context["possible_elements"])
        if not pair_infos:
            self._issue(
                issues,
                "warning",
                "pair_mapping",
                "No before/after pair manifest entry was found for this image.",
            )
        else:
            if len(pair_context["before_paths"]) > 1:
                self._issue(
                    issues,
                    "error",
                    "ambiguous_before_mapping",
                    "The same after image is mapped to multiple before images.",
                )
            if len(pair_infos) > len(pair_context["before_paths"]):
                self._issue(
                    issues,
                    "warning",
                    "duplicate_pair",
                    f"This after image occurs {len(pair_infos)} times in pair manifests.",
                )

            expected_fields = {
                POSSIBLE_ELEMENT_FIELDS[element]
                for element in possible_elements
                if element in POSSIBLE_ELEMENT_FIELDS
            }
            if possible_elements.intersection(MAIN_DISH_ELEMENTS):
                expected_fields.add("pct_chicken_rice_veg")
            present_fields = set(numbers) | set(choices)
            missing_fields = sorted(expected_fields - present_fields)
            unexpected_fields = sorted(
                (present_fields & KNOWN_CONSUMPTION_FIELDS) - expected_fields
            )
            if missing_fields:
                self._issue(
                    issues,
                    "warning",
                    "missing_expected_fields",
                    "Tray configuration expects missing field(s): " + ", ".join(missing_fields),
                )
            if unexpected_fields:
                self._issue(
                    issues,
                    "warning",
                    "unexpected_fields",
                    "Annotation contains field(s) not expected for this tray: "
                    + ", ".join(unexpected_fields),
                )

            primary_before = pair_context["primary_before"]
            if primary_before is not None and self.mask_cache_dir.exists():
                before_stem = suffix_stem(primary_before.name)
                before_classes = self._classes_for_stem(before_stem)
                after_classes = self._classes_for_stem(stem)
                allowed_classes = {
                    POSSIBLE_ELEMENT_CLASSES[element]
                    for element in possible_elements
                    if element in POSSIBLE_ELEMENT_CLASSES
                }
                new_classes = after_classes - before_classes
                unexpected_new = sorted(new_classes - allowed_classes)
                expected_new = sorted(new_classes & allowed_classes)
                if unexpected_new:
                    self._issue(
                        issues,
                        "warning",
                        "unexpected_class_appears_after",
                        "Detector found class(es) only after consumption and not in the tray "
                        "configuration: " + ", ".join(unexpected_new),
                    )
                if expected_new:
                    self._issue(
                        issues,
                        "warning",
                        "class_appears_only_after",
                        "Detector found class(es) after but not before: " + ", ".join(expected_new),
                    )

                for metric_name, metric_value in valid_metric_values.items():
                    target_classes = set(self._model_classes(metric_name))
                    if not before_classes.intersection(target_classes):
                        self._issue(
                            issues,
                            "warning",
                            "target_missing_before",
                            f"Detector found no before-mask for {metric_name} "
                            f"({sorted(target_classes)}).",
                        )
                    if metric_value < 100 and not after_classes.intersection(target_classes):
                        self._issue(
                            issues,
                            "warning",
                            "target_missing_after",
                            f"{metric_name}={metric_value:g}, but the detector found no matching "
                            "class after consumption.",
                        )

                for field_name, model_class in CHOICE_FIELD_TO_CLASS.items():
                    selected = self._selected_values(choices.get(field_name, []))
                    if selected == ["Not consumed"] and model_class not in after_classes:
                        self._issue(
                            issues,
                            "warning",
                            "not_consumed_but_missing_after",
                            f"{field_name} is 'Not consumed', but '{model_class}' was not "
                            "detected after consumption.",
                        )

        if self.after_image_paths.get(stem) is None:
            self._issue(
                issues,
                "error",
                "image_mapping",
                "No local consumed image was found for this task.",
            )

        order = {"error": 0, "quality": 1, "warning": 2}
        return sorted(issues, key=lambda issue: (order.get(issue["severity"], 9), issue["check"]))

    def summarize_tasks(self) -> List[Dict[str, Any]]:
        if self._summary_cache is not None:
            return self._summary_cache
        summary: List[Dict[str, Any]] = []
        for stem in self.stems:
            record = self.indexed[stem]
            issues = self.validate_record(stem, record)
            severity_counts = Counter(issue["severity"] for issue in issues)
            quality_flags = [
                issue["message"] for issue in issues if issue["check"] == "quality_flag"
            ]
            summary.append(
                {
                    "stem": stem,
                    "task_id": record.get("task_id"),
                    "source_file": record.get("source_file"),
                    "has_local_image": self.after_image_paths.get(stem) is not None,
                    "issue_count": len(issues),
                    "error_count": severity_counts["error"],
                    "warning_count": severity_counts["warning"],
                    "quality_count": severity_counts["quality"],
                    "quality_flags": quality_flags,
                }
            )
        self._summary_cache = summary
        return summary

    def meta(self) -> Dict[str, Any]:
        tasks = self.summarize_tasks()
        return {
            "task_count": len(tasks),
            "problem_count": sum(task["issue_count"] > 0 for task in tasks),
            "quality_count": sum(task["quality_count"] > 0 for task in tasks),
            "error_count": sum(task["error_count"] > 0 for task in tasks),
            "warning_count": sum(task["warning_count"] > 0 for task in tasks),
            "dataset_warnings": self.dataset_warnings,
        }

    def task_detail(self, stem: str) -> Dict[str, Any]:
        record = self.indexed[stem]
        after_path = self.after_image_paths.get(stem)
        pair_context = self._pair_context(stem)
        before_path = pair_context["primary_before"]

        numbers = [
            {"from_name": field, "value": values[0] if values else None}
            for field, values in record.get("numbers", {}).items()
        ]
        choices: List[Dict[str, Any]] = []
        for field, values in record.get("choices", {}).items():
            selected = values[0] if values else []
            options = list(self.choice_options.get(field, []))
            for value in selected:
                if value not in options:
                    options.append(value)
            choices.append(
                {
                    "from_name": field,
                    "values": selected,
                    "options": options,
                    "multiple": field == "quality_flags",
                }
            )
        if "quality_flags" not in record.get("choices", {}):
            choices.append(
                {
                    "from_name": "quality_flags",
                    "values": [],
                    "options": list(self.choice_options.get("quality_flags", [])),
                    "multiple": True,
                }
            )

        return {
            "stem": stem,
            "task_id": record.get("task_id"),
            "source_file": record.get("source_file"),
            "image_url": f"/image/after/{quote(stem)}" if after_path else None,
            "after_image_url": f"/image/after/{quote(stem)}" if after_path else None,
            "before_image_url": f"/image/before/{quote(stem)}" if before_path else None,
            "numbers": numbers,
            "choices": choices,
            "issues": self.validate_record(stem, record),
            "possible_elements": pair_context["possible_elements"],
            "splits": pair_context["splits"],
            "categories": pair_context["categories"],
        }

    def update_task(self, stem: str, payload: Dict[str, Any]) -> None:
        record = self.indexed[stem]
        numbers = record.setdefault("numbers", {})
        choices = record.setdefault("choices", {})
        number_updates = payload.get("numbers", {})
        choice_updates = payload.get("choices", {})
        number_deletes = set(payload.get("delete_numbers", []))

        for field in number_deletes:
            numbers.pop(field, None)
        for field, raw_val in number_updates.items():
            if field in number_deletes:
                continue
            try:
                numbers[field] = [float(raw_val)]
            except (TypeError, ValueError):
                continue
        for field, raw_vals in choice_updates.items():
            if isinstance(raw_vals, list):
                choices[field] = [[str(value) for value in raw_vals]]
            elif isinstance(raw_vals, str):
                choices[field] = [[raw_vals]] if raw_vals else [[]]
        self._summary_cache = None

    def save(self) -> Path:
        self.data["indexed_by_image_stem"] = {
            stem: [record] + self.extra_records_by_stem.get(stem, [])
            for stem, record in self.indexed.items()
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return self.output_path

    def image_path_for_stem(self, stem: str, side: str = "after") -> Path | None:
        if side == "after":
            return self.after_image_paths.get(stem)
        pair_context = self._pair_context(stem)
        return pair_context["primary_before"]


_LEGACY_HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Consumption Review UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; }
    .layout { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
    .left { border-right: 1px solid #ddd; overflow: auto; padding: 12px; }
    .right { padding: 12px; overflow: auto; }
    .task { padding: 8px; border: 1px solid #ddd; margin-bottom: 8px; cursor: pointer; }
    .task.bad { border-left: 4px solid #c62828; }
    .task.good { border-left: 4px solid #2e7d32; }
    .row { margin-bottom: 8px; }
    label { display: inline-block; min-width: 180px; }
    input[type=number] { min-width: 180px; }
    .delete-toggle { margin-left: 10px; font-size: 12px; color: #444; min-width: 0; }
    .choice-options { display: flex; flex-wrap: wrap; gap: 10px; }
    .choice-options label { min-width: 0; font-weight: normal; }
    .issues { background: #fff7f7; border: 1px solid #f5c2c2; padding: 10px; margin: 10px 0; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 10px; }
    img { max-width: 100%; display: block; }
    .small { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div class=\"layout\">
    <div class="left">
      <div class="toolbar">
        <button onclick="refreshTasks()">Reload</button>
        <button onclick="saveAll()">Save JSON</button>
      </div>
      <div class="row" style="padding: 0 4px; font-size: 13px;">
        <label style="min-width: unset; cursor: pointer;">
          <input type="checkbox" id="showOnlyIssues" onchange="renderTaskList()" /> Show only tasks with issues
        </label>
      </div>
      <div id="taskList"></div>
    </div>
    <div class=\"right\">
      <h2 id=\"title\">Select a task</h2>
      <div id=\"meta\" class=\"small\"></div>
      <div id=\"issues\"></div>
      <img id=\"img\" alt=\"consumed image\" />
      <h3>Numbers</h3>
      <div class=\"small\">Tip: tick Delete to remove a numeric field entry from this task.</div>
      <div id=\"numbers\"></div>
      <h3>Choices</h3>
      <div id=\"choices\"></div>
      <div class=\"toolbar\">
        <button onclick=\"saveTask()\">Save Task Changes</button>
      </div>
    </div>
  </div>

  <script>
    let tasks = [];
    let current = null;

    async function api(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function refreshTasks() {
      tasks = await api('/api/tasks');
      renderTaskList();
    }

    function renderTaskList() {
      const list = document.getElementById('taskList');
      list.innerHTML = '';
      const showOnlyIssues = document.getElementById('showOnlyIssues').checked;

      for (const t of tasks) {
        if (showOnlyIssues && t.issue_count <= 0) {
          continue;
        }
        const el = document.createElement('div');
        el.className = 'task ' + (t.issue_count > 0 ? 'bad' : 'good');
        el.innerHTML = `<b>${t.stem}</b><br>task=${t.task_id} | ${t.source_file || '(no file)'}<br>issues=${t.issue_count}`;
        el.onclick = () => loadTask(t.stem);
        list.appendChild(el);
      }
    }

    async function loadTask(stem) {
      current = await api('/api/task/' + encodeURIComponent(stem));
      document.getElementById('title').textContent = `${current.stem} (task ${current.task_id})`;
      document.getElementById('meta').textContent = `source: ${current.source_file || '(missing)'}`;

      const issues = document.getElementById('issues');
      if (current.issues.length) {
        issues.className = 'issues';
        issues.innerHTML = '<b>Issues</b><br>' + current.issues.map(i => `${i.check}: ${i.message}`).join('<br>');
      } else {
        issues.className = '';
        issues.innerHTML = '<span class="small">No validation issues for this task.</span>';
      }

      renderNumbers(current.numbers);
      renderChoices(current.choices);

      const img = document.getElementById('img');
      if (current.image_url) {
        img.src = current.image_url + '?t=' + Date.now();
      } else {
        img.removeAttribute('src');
      }
    }

    function renderNumbers(numbers) {
      const wrap = document.getElementById('numbers');
      wrap.innerHTML = '';
      for (const n of numbers) {
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<label>${n.from_name}</label><input type=\"number\" step=\"1\" id=\"num_${n.from_name}\" value=\"${n.value ?? ''}\" /><label class=\"delete-toggle\"><input type=\"checkbox\" id=\"del_${n.from_name}\" /> Delete</label>`;
        wrap.appendChild(row);
      }
    }

    function renderChoices(choices) {
      const wrap = document.getElementById('choices');
      wrap.innerHTML = '';
      for (const c of choices) {
        const row = document.createElement('div');
        row.className = 'row';
        const optionsHtml = c.options.map(o => {
          const checked = c.values.includes(o) ? 'checked' : '';
          const id = `choice_${c.from_name}_${o.replace(/[^a-zA-Z0-9]/g, '_')}`;
          return `<label><input type=\"checkbox\" id=\"${id}\" value=\"${o}\" ${checked} /> ${o}</label>`;
        }).join('');
        row.innerHTML = `<label>${c.from_name}</label><div class=\"choice-options\" data-field=\"${c.from_name}\">${optionsHtml}</div>`;
        wrap.appendChild(row);
      }
    }

    async function saveTask() {
      if (!current) return;
      const numbers = {};
      const delete_numbers = [];
      for (const n of current.numbers) {
        const del = document.getElementById('del_' + n.from_name);
        if (del && del.checked) {
          delete_numbers.push(n.from_name);
          continue;
        }
        const el = document.getElementById('num_' + n.from_name);
        if (!el) continue;
        numbers[n.from_name] = el.value;
      }

      const choices = {};
      for (const c of current.choices) {
        const group = document.querySelector(`.choice-options[data-field="${c.from_name}"]`);
        if (!group) continue;
        const checked = Array.from(group.querySelectorAll('input[type=checkbox]:checked')).map(i => i.value);
        choices[c.from_name] = checked;
      }

      await api('/api/task/' + encodeURIComponent(current.stem), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({numbers, choices, delete_numbers}),
      });

      await loadTask(current.stem);
      await refreshTasks();
    }

    async function saveAll() {
      const res = await api('/api/save', { method: 'POST' });
      alert('Saved to: ' + res.path);
    }

    refreshTasks();
  </script>
</body>
</html>
"""


HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consumption Dataset Review</title>
<style>
:root{--bg:#f5f7fa;--panel:#fff;--line:#dfe4ea;--text:#17212b;--muted:#667085;--blue:#2457d6;--error:#b42318;--warn:#b54708;--quality:#7a3e9d;--good:#067647;--error-bg:#fff1f0;--warn-bg:#fff7ed;--quality-bg:#f8f0fc;--good-bg:#ecfdf3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}button,input,select{font:inherit}button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:8px 12px;cursor:pointer}button:hover{border-color:#98a2b3}button.primary{color:#fff;background:var(--blue);border-color:var(--blue)}
.app{display:grid;grid-template-columns:390px minmax(0,1fr);height:100vh}.sidebar{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}.sidebar-head{padding:16px;border-bottom:1px solid var(--line)}h1{font-size:20px;margin:0 0 12px}h2{margin:0;font-size:21px;overflow-wrap:anywhere}h3{margin:20px 0 10px;font-size:16px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:12px}.stat{background:var(--bg);border-radius:7px;padding:7px;text-align:center}.stat b{display:block;font-size:17px}.stat span,.small{color:var(--muted);font-size:11px}.toolbar,.detail-actions{display:flex;flex-wrap:wrap;gap:7px}.filters{display:grid;grid-template-columns:1fr 135px;gap:7px;margin-top:9px}.filters input,.filters select{width:100%;border:1px solid var(--line);border-radius:7px;padding:8px;background:#fff}.dataset-warnings{padding:0 16px}.dataset-warning{margin-top:10px;padding:9px;border:1px solid #fedf89;background:#fffaeb;border-radius:7px;color:#7a2e0e;font-size:12px}
.task-list{padding:10px;overflow:auto;min-height:0}.task{border:1px solid var(--line);border-left:4px solid #98a2b3;background:#fff;border-radius:8px;padding:10px;margin-bottom:8px;cursor:pointer}.task:hover,.task.active{border-color:var(--blue);box-shadow:0 1px 4px #10182818}.task.has-error{border-left-color:var(--error)}.task.has-quality{border-left-color:var(--quality)}.task.has-warning{border-left-color:var(--warn)}.task.clean{border-left-color:var(--good)}.task-title{font-weight:650;overflow-wrap:anywhere}.task-meta{color:var(--muted);font-size:12px;margin-top:3px}.badges,.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.badge,.chip{display:inline-block;border-radius:999px;padding:2px 7px;font-size:11px}.badge.error{color:var(--error);background:var(--error-bg)}.badge.warning{color:var(--warn);background:var(--warn-bg)}.badge.quality{color:var(--quality);background:var(--quality-bg)}.badge.clean{color:var(--good);background:var(--good-bg)}.chip{background:#eef2f6;color:#344054}
.main{overflow:auto;min-width:0}.content{max-width:1500px;margin:0 auto;padding:22px 26px 60px}.detail-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.muted{color:var(--muted)}.issues{display:grid;gap:8px;margin:18px 0}.issue{border:1px solid;border-radius:8px;padding:10px 12px}.issue.error{color:#7a271a;background:var(--error-bg);border-color:#fecdca}.issue.warning{color:#7a2e0e;background:var(--warn-bg);border-color:#fed7aa}.issue.quality{color:#5b2a72;background:var(--quality-bg);border-color:#e9d5f2}.issue b{margin-right:6px}.all-good{color:var(--good);background:var(--good-bg);border:1px solid #abefc6;border-radius:8px;padding:10px 12px}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.image-card{margin:0;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden}.image-card figcaption{padding:8px 11px;font-weight:650;border-bottom:1px solid var(--line)}.image-frame{min-height:220px;background:#eef1f5;display:grid;place-items:center}.image-frame img{display:block;max-width:100%;max-height:68vh;object-fit:contain}.image-missing{color:var(--muted);padding:30px}
.form-section{background:#fff;border:1px solid var(--line);border-radius:9px;padding:4px 16px 14px;margin-top:14px}.field-row{display:grid;grid-template-columns:230px minmax(0,1fr);gap:16px;padding:11px 0;border-bottom:1px solid #eef0f3}.field-row:last-child{border-bottom:0}.field-name{font-weight:600;overflow-wrap:anywhere}input[type=number]{width:160px;border:1px solid var(--line);border-radius:7px;padding:8px}.delete-toggle{margin-left:12px;color:var(--muted);font-size:12px}.choice-options{display:flex;flex-wrap:wrap;gap:8px 16px}.choice-options label{cursor:pointer}.save-bar{position:sticky;bottom:0;display:flex;align-items:center;gap:10px;background:#fffffff2;border:1px solid var(--line);border-radius:9px;padding:10px;margin-top:15px;box-shadow:0 -4px 18px #10182812}.empty{display:grid;place-items:center;min-height:60vh;color:var(--muted);text-align:center}@media(max-width:950px){.app{grid-template-columns:320px 1fr}.images{grid-template-columns:1fr}}@media(max-width:700px){.app{display:block;height:auto}.sidebar{max-height:55vh;border-right:0;border-bottom:1px solid var(--line)}.content{padding:16px}.field-row{grid-template-columns:1fr;gap:7px}}
</style></head><body><div class="app">
<aside class="sidebar"><div class="sidebar-head"><h1>Consumption review</h1><div class="stats" id="stats"></div><div class="toolbar"><button onclick="refreshTasks()">Reload</button><button class="primary" onclick="saveAll()">Write JSON</button></div><div class="filters"><input id="search" type="search" placeholder="Search ID or filename" oninput="renderTaskList()"><select id="filter" onchange="renderTaskList()"><option value="problems">All problems</option><option value="quality">Quality flags</option><option value="errors">Errors</option><option value="warnings">Warnings</option><option value="clean">Clean</option><option value="all">All images</option></select></div></div><div id="datasetWarnings" class="dataset-warnings"></div><div id="taskList" class="task-list"></div></aside>
<main class="main"><div id="empty" class="empty"><div><h2>Select an image</h2><p>Quality-flagged and invalid records are listed first.</p></div></div><div id="detail" class="content" hidden><div class="detail-head"><div><h2 id="title"></h2><div id="meta" class="muted"></div><div id="chips" class="chips"></div></div><div class="detail-actions"><button onclick="moveTask(-1)">← Previous</button><button onclick="moveTask(1)">Next →</button></div></div><div id="issues" class="issues"></div><div class="images"><figure class="image-card"><figcaption>Before / unconsumed</figcaption><div id="beforeFrame" class="image-frame"></div></figure><figure class="image-card"><figcaption>After / consumed</figcaption><div id="afterFrame" class="image-frame"></div></figure></div><div class="form-section"><h3>Percentages</h3><div class="small">Values must be between 0 and 100. Delete removes an incorrect field.</div><div id="numbers"></div></div><div class="form-section"><h3>Choices and quality flags</h3><div id="choices"></div></div><div class="save-bar"><button class="primary" onclick="saveTask()">Apply task changes</button><span id="saveStatus" class="small">Changes are held in memory until Write JSON is clicked.</span></div></div></main></div>
<script>
let tasks=[],current=null,meta={};
async function api(url,options){const res=await fetch(url,options);if(!res.ok)throw new Error(await res.text());return res.json()}
function esc(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function visibleTasks(){const filter=document.getElementById('filter').value,query=document.getElementById('search').value.trim().toLowerCase();return tasks.filter(t=>{const search=!query||`${t.stem} ${t.task_id} ${t.source_file||''}`.toLowerCase().includes(query);const kind=filter==='all'||(filter==='problems'&&t.issue_count>0)||(filter==='quality'&&t.quality_count>0)||(filter==='errors'&&t.error_count>0)||(filter==='warnings'&&t.warning_count>0)||(filter==='clean'&&t.issue_count===0);return search&&kind})}
async function refreshTasks(){tasks=await api('/api/tasks');meta=await api('/api/meta');tasks.sort((a,b)=>(b.error_count-a.error_count)||(b.quality_count-a.quality_count)||(b.warning_count-a.warning_count)||a.stem.localeCompare(b.stem));renderStats();renderWarnings();renderTaskList()}
function renderStats(){document.getElementById('stats').innerHTML=`<div class="stat"><b>${meta.task_count??0}</b><span>images</span></div><div class="stat"><b>${meta.problem_count??0}</b><span>problems</span></div><div class="stat"><b>${meta.quality_count??0}</b><span>quality</span></div>`}
function renderWarnings(){document.getElementById('datasetWarnings').innerHTML=(meta.dataset_warnings||[]).map(w=>`<div class="dataset-warning">${esc(w)}</div>`).join('')}
function badge(label,count,kind){return count?`<span class="badge ${kind}">${count} ${label}</span>`:''}
function renderTaskList(){const list=document.getElementById('taskList'),visible=visibleTasks();list.innerHTML='';for(const t of visible){const el=document.createElement('div'),state=t.error_count?'has-error':t.quality_count?'has-quality':t.warning_count?'has-warning':'clean';el.className=`task ${state} ${current&&current.stem===t.stem?'active':''}`;const quality=t.quality_flags.length?`<div class="task-meta">${esc(t.quality_flags.join(' · '))}</div>`:'';el.innerHTML=`<div class="task-title">${esc(t.stem)}</div><div class="task-meta">task ${esc(t.task_id??'?')}</div>${quality}<div class="badges">${badge('error',t.error_count,'error')}${badge('quality',t.quality_count,'quality')}${badge('warning',t.warning_count,'warning')}${!t.issue_count?'<span class="badge clean">clean</span>':''}</div>`;el.onclick=()=>loadTask(t.stem);list.appendChild(el)}if(!visible.length)list.innerHTML='<div class="empty" style="min-height:180px">No images match this filter.</div>'}
function setImage(frameId,url,alt){document.getElementById(frameId).innerHTML=url?`<a href="${esc(url)}" target="_blank"><img src="${esc(url)}?t=${Date.now()}" alt="${esc(alt)}"></a>`:'<div class="image-missing">Image not available</div>'}
async function loadTask(stem){current=await api('/api/task/'+encodeURIComponent(stem));document.getElementById('empty').hidden=true;document.getElementById('detail').hidden=false;document.getElementById('title').textContent=current.stem;document.getElementById('meta').textContent=`Task ${current.task_id??'?'} · ${current.source_file||'missing source filename'}`;const values=[...(current.categories||[]),...(current.splits||[]),...(current.possible_elements||[])];document.getElementById('chips').innerHTML=values.map(v=>`<span class="chip">${esc(v)}</span>`).join('');renderIssues(current.issues);setImage('beforeFrame',current.before_image_url,'before image');setImage('afterFrame',current.after_image_url,'after image');renderNumbers(current.numbers);renderChoices(current.choices);renderTaskList();document.getElementById('saveStatus').textContent='Changes are held in memory until Write JSON is clicked.'}
function renderIssues(issues){document.getElementById('issues').innerHTML=issues.length?issues.map(i=>`<div class="issue ${esc(i.severity)}"><b>${esc(i.severity.toUpperCase())}</b><span>${esc(i.message)}</span><div class="small">${esc(i.check)}</div></div>`).join(''):'<div class="all-good">No validation or quality problems found.</div>'}
function renderNumbers(numbers){const wrap=document.getElementById('numbers');wrap.innerHTML='';for(const n of numbers){const row=document.createElement('div');row.className='field-row';row.innerHTML=`<div class="field-name">${esc(n.from_name)}</div><div><input type="number" min="0" max="100" step="1" id="num_${esc(n.from_name)}" value="${esc(n.value??'')}"><label class="delete-toggle"><input type="checkbox" id="del_${esc(n.from_name)}"> Delete field</label></div>`;wrap.appendChild(row)}if(!numbers.length)wrap.innerHTML='<p class="small">No numeric fields.</p>'}
function renderChoices(choices){const wrap=document.getElementById('choices');wrap.innerHTML='';for(const c of choices){const row=document.createElement('div');row.className='field-row';const type=c.multiple?'checkbox':'radio',qualityChange=c.from_name==='quality_flags'?'onchange="normalizeQualityChoice(this)"':'';const options=c.options.map((o,i)=>`<label><input type="${type}" name="choice_${esc(c.from_name)}" id="choice_${esc(c.from_name)}_${i}" value="${esc(o)}" ${c.values.includes(o)?'checked':''} ${qualityChange}> ${esc(o)}</label>`).join('');row.innerHTML=`<div class="field-name">${esc(c.from_name)}</div><div class="choice-options" data-field="${esc(c.from_name)}">${options}</div>`;wrap.appendChild(row)}}
function normalizeQualityChoice(changed){if(!changed.checked)return;const group=changed.closest('.choice-options'),clean='No issues — image pair is clean';for(const input of group.querySelectorAll('input')){if(input===changed)continue;if(changed.value===clean||input.value===clean)input.checked=false}}
function moveTask(delta){const visible=visibleTasks();if(!visible.length)return;const index=Math.max(0,visible.findIndex(t=>current&&t.stem===current.stem));loadTask(visible[(index+delta+visible.length)%visible.length].stem)}
async function saveTask(){if(!current)return;const numbers={},delete_numbers=[];for(const n of current.numbers){const del=document.getElementById('del_'+n.from_name);if(del&&del.checked){delete_numbers.push(n.from_name);continue}const input=document.getElementById('num_'+n.from_name);if(input)numbers[n.from_name]=input.value}const choices={};for(const c of current.choices){const group=document.querySelector(`.choice-options[data-field="${c.from_name}"]`);if(group)choices[c.from_name]=Array.from(group.querySelectorAll('input:checked')).map(i=>i.value)}await api('/api/task/'+encodeURIComponent(current.stem),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({numbers,choices,delete_numbers})});const stem=current.stem;await refreshTasks();await loadTask(stem);document.getElementById('saveStatus').textContent='Task changes applied. Click Write JSON to persist them.'}
async function saveAll(){const result=await api('/api/save',{method:'POST'});document.getElementById('saveStatus').textContent='Saved to '+result.path;alert('Saved to: '+result.path)}
refreshTasks().catch(error=>alert(error.message));
</script></body></html>"""


def build_handler(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/tasks":
                self._send_json(session.summarize_tasks())
                return

            if self.path == "/api/meta":
                self._send_json(session.meta())
                return

            if self.path.startswith("/api/task/"):
                try:
                    stem = unquote(self.path.split("/api/task/", 1)[1])
                    self._send_json(session.task_detail(stem))
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            if self.path.startswith("/image/"):
                try:
                    image_key = self.path.split("/image/", 1)[1].split("?")[0]
                    parts = image_key.split("/", 1)
                    if len(parts) == 2 and parts[0] in {"before", "after"}:
                        side, encoded_stem = parts
                    else:
                        side, encoded_stem = "after", image_key
                    stem = unquote(encoded_stem)
                    path = session.image_path_for_stem(stem, side=side)
                    if path is None or not path.exists():
                        self._send_text("Image not found", status=404)
                        return
                    data = path.read_bytes()
                    ext = path.suffix.lower()
                    mime = "image/jpeg"
                    if ext == ".png":
                        mime = "image/png"
                    elif ext == ".webp":
                        mime = "image/webp"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            self._send_text("Not found", status=404)

        def do_POST(self) -> None:
            if self.path.startswith("/api/task/"):
                try:
                    stem = unquote(self.path.split("/api/task/", 1)[1])
                    payload = self._read_json_body()
                    session.update_task(stem, payload)
                    self._send_json({"ok": True})
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            if self.path == "/api/save":
                try:
                    out_path = session.save()
                    self._send_json({"ok": True, "path": str(out_path)})
                except Exception as exc:
                    self._send_text(str(exc), status=500)
                return

            self._send_text("Not found", status=404)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep terminal output focused on review progress.
            return

    return Handler


def main() -> int:
    args = parse_args()
    session = ReviewSession(
        input_path=args.input.resolve(),
        images_dir=args.images_dir.resolve(),
        output_path=args.output.resolve() if args.output else None,
        before_images_dir=args.before_images_dir.resolve(),
        pairs_dir=args.pairs_dir.resolve(),
        mask_cache_dir=args.mask_cache_dir.resolve(),
    )

    handler = build_handler(session)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Review UI running at http://{args.host}:{args.port}")
    print(f"Input index  : {session.input_path}")
    print(f"Images dir   : {session.images_dir}")
    print(f"Pairs dir    : {session.pairs_dir}")
    print(f"Mask cache   : {session.mask_cache_dir}")
    print(f"Save output  : {session.output_path}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
