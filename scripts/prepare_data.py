import os
import shutil
import random
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# Edit these paths and settings to match your setup
# ══════════════════════════════════════════════════════════════════════════════

# Every source directory is treated as one combined pool of images before
# splitting — i.e. train/val/test are drawn from ALL sources together, not
# split independently per source.
#
# Each Label Studio project exports its OWN classes.txt, and the numeric
# class index inside that project's YOLO .txt label files only makes sense
# relative to ITS OWN classes.txt (line 0 = class 0, line 1 = class 1, ...).
# Since different projects can (and do) list their classes in different
# orders / with different names, we read each source's classes.txt
# separately and remap by NAME (see NAME_TO_FINAL below) rather than by a
# single hardcoded numeric table.
DATA_SOURCES = [
    {
        "images": Path("data/images"),
        "labels": Path("data/labels"),
        "classes": Path("data/classes.txt"),
    },
    {
        "images": Path("data_consumed/images"),
        "labels": Path("data_consumed/labels"),
        "classes": Path("data_consumed/classes.txt"),
    },
]

OUTPUT_DIR = Path("data_with_splits")  # where the finished split will be written

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

SEED        = 42    # change for a different random split; keep fixed for reproducibility
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test ratio is whatever remains (0.15)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASS MAPPING (by name, not by raw index)
#
# Keys   = class name exactly as it appears in EITHER source's classes.txt
# Values = final class name to use in the merged dataset (must match an
#          entry in CLASS_NAMES below), or None to drop that class entirely.
#
# Every name found across BOTH classes.txt files should appear here once.
# Non-object entries (Blurry, Consumed, Not consumed, etc.) come from the
# Choices fields used for the consumption annotations in the same Label
# Studio project — they never have polygon geometry, so they map to None
# and are silently dropped when remapping segmentation labels.
# ══════════════════════════════════════════════════════════════════════════════

NAME_TO_FINAL = {
    # --- non-object / choice classes: never appear as real polygon regions ---
    "Blurry": None,
    "Consumed": None,
    "Food rearranged significantly": None,
    "Lighting inconsistency vs other images": None,
    "No issues — image pair is clean": None,
    "Non-edible residuals present (bones, core, crumbs only)": None,
    "Not consumed": None,
    "Obstacle obscuring food (cutlery, napkin, foil)": None,
    "Odd tray angle": None,
    "Shadow on tray": None,
    "Tray partially out of frame": None,

    # --- food / object classes ---
    "bread_roll": "bread_roll",
    "broccoli": "broccoli",
    "brownie": "chocolate_cake",                 # MERGE into chocolate_cake
    "butter": "butter",
    "carrots": "carrots",
    "cherry_jam": "cherry_jam",
    "chicken": "chicken",
    "chocolate_cake": "chocolate_cake",
    "coffee_cup": "coffee",                      # RENAME
    "cola": "cola",
    "cola_can": "cola",                          # RENAME
    "cookie": "cookie",
    "fish_salmon": "fish_salmon",
    "fruit_salad": "fruit_salad",
    "honey": "honey",
    "orange_juice_bottle": "orange_juice",        # RENAME
    "pasta_pesto": "pasta_pesto",                 # NOTE: 0 => was skipped given time constraints 
    "plum_jam": "plum_jam",
    "rice": "rice",
    "main_salad": "main_salad",                   # ALREADY CANONICAL
    "salad_main": "main_salad",                   # RENAME
    "side_salad": "side_salad",
    "tea_cup": "tea",                             # RENAME
    "vanilla_pudding_with_fruits": "vanilla_pudding_with_fruits",
    "water_bottle": "water",                      # RENAME
    "wrap_half_1": "wrap_half_1",
    "wrap_half_2": "wrap_half_2",
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FINAL CLASS NAMES
# Must be in new-index order (index 0 first, last index last)
# Only edit these if you want to further rename something later
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "bread_roll",                   # 0
    "broccoli",                     # 1
    "butter",                       # 2
    "carrots",                      # 3
    "cherry_jam",                   # 4
    "chicken",                      # 5
    "chocolate_cake",               # 6
    "coffee",                       # 7
    "cola",                         # 8
    "cookie",                       # 9
    "fish_salmon",                  # 10
    "fruit_salad",                  # 11
    "honey",                        # 12
    "orange_juice",                 # 13
    "pasta_pesto",                  # 14
    "plum_jam",                     # 15
    "rice",                         # 16
    "main_salad",                   # 17
    "side_salad",                   # 18
    "tea",                          # 19
    "vanilla_pudding_with_fruits",  # 20
    "water",                        # 21
    "wrap_half_1",                  # 22
    "wrap_half_2",                  # 23
]

FINAL_NAME_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORE LOGIC (you don't need to edit below this line)
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names(classes_path: Path):
    """
    Reads a Label Studio classes.txt file. Line N (0-based) is the name
    used for class index N in that project's exported YOLO .txt labels.
    """
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path.resolve()}")
    lines = classes_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines]


def remap_label_file(src: Path, local_class_names):
    """
    Reads one YOLO .txt label file and returns remapped lines, translating
    this project's local class index -> class name (via local_class_names)
    -> final class name (via NAME_TO_FINAL) -> final class index.
    - Lines whose local index has no known name, whose name has no entry
      in NAME_TO_FINAL, or whose entry maps to None are dropped (with a
      warning, except for the expected None/choice-class case).
    - Bounding box / polygon coordinates are kept exactly as-is;
      only the class index at position 0 changes.
    """
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_idx = int(parts[0])

        if old_idx >= len(local_class_names):
            print(f"  ⚠ Index {old_idx} in {src.name} has no matching line in "
                  f"this source's classes.txt — skipping line")
            continue

        name = local_class_names[old_idx]

        if name not in NAME_TO_FINAL:
            print(f"  ⚠ Unknown class name '{name}' (index {old_idx}) in {src.name} — "
                  f"skipping line. Add it to NAME_TO_FINAL if this is expected.")
            continue

        final_name = NAME_TO_FINAL[name]
        if final_name is None:
            # Expected: a Choices-field class (e.g. "Consumed"), never a real region.
            continue

        if final_name not in FINAL_NAME_TO_INDEX:
            print(f"  ⚠ '{name}' maps to final name '{final_name}', which is not in "
                  f"CLASS_NAMES — skipping line")
            continue

        new_cls = FINAL_NAME_TO_INDEX[final_name]
        lines_out.append(f"{new_cls} {' '.join(parts[1:])}")
    return lines_out


def collect_images():
    """
    Walks every entry in DATA_SOURCES, loads each source's own classes.txt,
    and returns a combined list of (image_path, source) tuples — one per
    image, remembering which source dict (images/labels/classes +
    class_names) it belongs to. Raises if the same filename appears in more
    than one source, since that would silently collide in the output split.
    """
    all_images = []
    seen_in = {}  # filename -> images dir it was first seen in

    for source in DATA_SOURCES:
        images_dir = source["images"]
        classes_path = source["classes"]

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

        source["class_names"] = load_class_names(classes_path)
        unknown = [n for n in source["class_names"] if n and n not in NAME_TO_FINAL]
        if unknown:
            print(f"  ⚠ {classes_path} has {len(unknown)} name(s) with no NAME_TO_FINAL "
                  f"entry (lines using them will be skipped): {unknown}")

        found_here = [p for p in images_dir.iterdir() if p.suffix in IMG_EXTS]
        print(f"  found {len(found_here)} images in '{images_dir}' "
              f"(classes.txt: {len(source['class_names'])} names)")

        for p in found_here:
            if p.name in seen_in:
                raise ValueError(
                    f"Duplicate filename '{p.name}' found in both "
                    f"'{seen_in[p.name]}' and '{images_dir}'. "
                    "Rename one of the files before running the split, "
                    "otherwise one copy will silently overwrite the other."
                )
            seen_in[p.name] = images_dir
            all_images.append((p, source))

    return all_images


def copy_pair(img_path: Path, source: dict, split: str):
    """
    Copies one image to data_with_splits/<split>/images/
    and writes its remapped label to data_with_splits/<split>/labels/
    """
    label_dir = source["labels"]
    label_path = label_dir / img_path.with_suffix(".txt").name

    img_out = OUTPUT_DIR / split / "images" / img_path.name
    lbl_out = OUTPUT_DIR / split / "labels" / label_path.name

    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, img_out)

    if label_path.exists():
        remapped = remap_label_file(label_path, source["class_names"])
        lbl_out.write_text("\n".join(remapped), encoding="utf-8")
    else:
        print(f"  ⚠ No label file found for {img_path.name} — writing empty label")
        lbl_out.write_text("", encoding="utf-8")


def verify_mapping():
    """Sanity check: make sure every non-None NAME_TO_FINAL target exists in CLASS_NAMES."""
    targets = {v for v in NAME_TO_FINAL.values() if v is not None}
    missing = sorted(targets - set(CLASS_NAMES))
    if missing:
        raise ValueError(
            f"NAME_TO_FINAL points to name(s) not present in CLASS_NAMES: {missing}. "
            "Add them to CLASS_NAMES or fix the mapping."
        )
    print(f"✓ Mapping verified — {len(CLASS_NAMES)} final classes, "
          f"{len(NAME_TO_FINAL)} known source names")


def main():
    verify_mapping()

    print(f"\nCollecting images from {len(DATA_SOURCES)} source dir(s)...")
    images = collect_images()  # list of (img_path, source)
    if not images:
        raise FileNotFoundError("No images found across any of the DATA_SOURCES dirs.")

    print(f"\nFound {len(images)} images total across all sources")

    # Shuffle deterministically (combined pool, so train/val/test are drawn
    # from both sources together rather than split independently per source)
    random.seed(SEED)
    random.shuffle(images)

    # Compute split sizes
    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    print(f"\nSplit sizes:")
    for split, imgs in splits.items():
        print(f"  {split:5s}: {len(imgs)} images")

    print(f"\nProcessing...")
    for split, imgs in splits.items():
        for img_path, source in imgs:
            copy_pair(img_path, source, split)
        print(f"  ✓ {split} done")

    # Write dataset.yaml
    yaml_path  = OUTPUT_DIR / "dataset.yaml"
    nc         = len(CLASS_NAMES)
    names_str  = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = f"""# Auto-generated by prepare_dataset.py
path: {OUTPUT_DIR.resolve()}

train: train/images
val:   val/images
test:  test/images

nc: {nc}
names:
{names_str}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n✓ dataset.yaml written to {yaml_path}")

    # Print class distribution for train set
    print("\nClass distribution in train set:")
    counts = [0] * len(CLASS_NAMES)
    for img_path, _source in splits["train"]:
        lbl = OUTPUT_DIR / "train" / "labels" / img_path.with_suffix(".txt").name
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    cls = int(line.split()[0])
                    counts[cls] += 1

    for i, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        bar  = "█" * (count // 10)
        warn = "  ⚠ LOW" if count < 50 else ""
        print(f"  {i:2d}  {name:<35} {count:4d}  {bar}{warn}")

    print("\nAll done! ✓")
    print(f"Your dataset is ready at: {OUTPUT_DIR.resolve()}")
    print(f"Train with: yolo segment train data={yaml_path} model=yolo11m-seg.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
