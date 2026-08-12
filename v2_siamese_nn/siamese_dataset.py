"""PyTorch Dataset for the before/after consumption Siamese network.

Expands each (before, after) pair from data_pairs_with_splits/*.json into
one sample per pct_* metric present on that pair (e.g. a pair with both
pct_rice and pct_chicken_rice_veg becomes 2 samples). For composite metrics
(e.g. pct_chicken_rice_veg), the mask is the UNION of all aliased classes
("main dish" = chicken + rice + carrots + broccoli together) — matches how
that single percentage was actually annotated on the tray.

Each sample's before/after image is returned as a 4-channel tensor: RGB +
a binary mask channel built from the cached YOLO detections for the target
class(es). If no matching detection exists in the AFTER image, the mask
channel is simply all zeros — the model still sees the full photo and can
learn "no detected mask + tray still visible" as a real (usually ~100%
consumed) signal, no fallback cropping needed. If no matching detection
exists in the BEFORE image, there's no valid baseline for that class on
that tray, so the sample is dropped.

Pairs flagged "Food rearranged significantly" in quality_flags are dropped
entirely — that flag specifically invalidates the assumption the food is
still roughly where it was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MASK_CACHE_DIR = Path("mask_cache")
IMG_SIZE = 384  # full tray image, not a tight crop — bigger than the usual 224
                # so small items are still legible; drop to 224 if VRAM-constrained

# Must stay in sync with METRIC_LABEL_ALIASES in scripts/prepare_dataset.py —
# these are the "main dish" style composite metrics whose mask is the union
# of several classes rather than one class matching the metric name directly.
METRIC_LABEL_ALIASES: Dict[str, List[str]] = {
    "pct_vanilla_pudding": ["vanilla_pudding_with_fruits"],
    "pct_salad_dish_main": ["main_salad"],
    "pct_chicken_rice_veg": ["chicken", "rice", "carrots", "broccoli"],
}

QUALITY_FLAG_EXCLUDES = {"Food rearranged significantly"}

# Choice fields that represent a binary consumed/not-consumed status for a
# single item (as opposed to e.g. quality_flags, which is metadata about the
# photo itself). "Not present" (or any other value) means that item wasn't
# even on this tray, so it's skipped rather than treated as 0.
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
CHOICE_VALUE_TO_TARGET: Dict[str, float] = {"Consumed": 100.0, "Not consumed": 0.0}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def suffix_stem(name: str) -> str:
    """Strips a Label-Studio-style hash prefix like 'a5e6c40__' from a filename stem."""
    stem = Path(name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def expected_polygon_labels(metric_name: str) -> List[str]:
    if metric_name in METRIC_LABEL_ALIASES:
        return METRIC_LABEL_ALIASES[metric_name]
    if metric_name.startswith("pct_"):
        return [metric_name[4:]]
    return [metric_name]


def _first_choice_list(choices: Dict[str, Any], field: str) -> List[str]:
    """choices[field] is stored as [[val, ...]] (outer list length 1) — unwrap it."""
    values = choices.get(field, [])
    return values[0] if values else []


class MaskCache:
    """Small in-memory cache so repeated __getitem__ calls for the same
    image (common — one 'before' image is reused across many pairs) don't
    re-read the same JSON file from disk every time."""

    def __init__(self, cache_dir: Path = MASK_CACHE_DIR):
        self.cache_dir = cache_dir
        self._loaded: Dict[str, List[Dict[str, Any]]] = {}

    def detections_for(self, image_path: str) -> List[Dict[str, Any]]:
        stem = suffix_stem(Path(image_path).name)
        if stem not in self._loaded:
            cache_path = self.cache_dir / f"{stem}.json"
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as fh:
                    self._loaded[stem] = json.load(fh)["detections"]
            else:
                self._loaded[stem] = []
        return self._loaded[stem]

    def has_any_class(self, image_path: str, classes: Sequence[str]) -> bool:
        return any(det["class"] in classes for det in self.detections_for(image_path))


def build_mask_array(detections: List[Dict[str, Any]], classes: Sequence[str], img_size: int) -> np.ndarray:
    """Rasterizes the union of every detection whose class is in `classes`
    into a binary (0/1) mask of shape (img_size, img_size)."""
    mask_img = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask_img)
    for det in detections:
        if det["class"] not in classes:
            continue
        points = [(x * img_size, y * img_size) for x, y in det["polygon"]]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.array(mask_img, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ConsumptionPairDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        mask_cache_dir: Path = MASK_CACHE_DIR,
        img_size: int = IMG_SIZE,
        train_mode: bool = False,
    ):
        self.img_size = img_size
        self.train_mode = train_mode
        self.mask_cache = MaskCache(mask_cache_dir)
        self.rgb_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with Path(manifest_path).open("r", encoding="utf-8") as fh:
            pairs = json.load(fh)

        self.samples: List[Dict[str, Any]] = []
        dropped_quality = 0
        dropped_no_before_mask = 0

        for pair in pairs:
            flags = _first_choice_list(pair.get("choices", {}), "quality_flags")
            if set(QUALITY_FLAG_EXCLUDES) & set(flags):
                dropped_quality += 1
                continue

            for metric_name, values in pair.get("numbers", {}).items():
                if not values:
                    continue
                target_pct = values[0]
                classes = expected_polygon_labels(metric_name)

                # No baseline on the "before" tray for this class -> can't
                # define consumption for it, drop this specific sample.
                if not self.mask_cache.has_any_class(pair["before"], classes):
                    dropped_no_before_mask += 1
                    continue

                self.samples.append(
                    {
                        "before": pair["before"],
                        "after": pair["after"],
                        "category": pair.get("category"),
                        "classes": classes,
                        "metric_name": metric_name,
                        "target_pct": target_pct,
                        "task": "regression",
                    }
                )

            for field, values in pair.get("choices", {}).items():
                if field not in CHOICE_FIELD_TO_CLASS:
                    continue  # not a consumption-status field (e.g. quality_flags)
                selected = values[0] if values else []
                if not selected:
                    continue
                value = selected[0]
                if value not in CHOICE_VALUE_TO_TARGET:
                    continue  # "Not present" (or unrecognized) -> not on this tray
                target_pct = CHOICE_VALUE_TO_TARGET[value]
                classes = [CHOICE_FIELD_TO_CLASS[field]]

                if not self.mask_cache.has_any_class(pair["before"], classes):
                    dropped_no_before_mask += 1
                    continue

                self.samples.append(
                    {
                        "before": pair["before"],
                        "after": pair["after"],
                        "category": pair.get("category"),
                        "classes": classes,
                        "metric_name": field,
                        "target_pct": target_pct,
                        "task": "classification",
                    }
                )

        print(f"ConsumptionPairDataset({manifest_path}): {len(self.samples)} samples "
              f"(dropped {dropped_quality} quality-flagged pairs, "
              f"{dropped_no_before_mask} samples with no 'before' mask)")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, image_path: str, classes: Sequence[str]) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        detections = self.mask_cache.detections_for(image_path)
        mask_arr = build_mask_array(detections, classes, self.img_size)

        if self.train_mode and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask_arr = np.fliplr(mask_arr).copy()

        img_tensor = transforms.functional.to_tensor(img)  # 3xHxW, [0,1]
        img_tensor = self.rgb_normalize(img_tensor)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # 1xHxW, {0,1}

        return torch.cat([img_tensor, mask_tensor], dim=0)  # 4xHxW

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        before_tensor = self._load_tensor(sample["before"], sample["classes"])
        after_tensor = self._load_tensor(sample["after"], sample["classes"])
        target = torch.tensor(sample["target_pct"] / 100.0, dtype=torch.float32)
        return before_tensor, after_tensor, target, sample["task"]