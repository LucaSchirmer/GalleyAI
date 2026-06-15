import os
import shutil
import random
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# Edit these paths and settings to match your setup
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR   = Path("data/images")      # folder containing your images
OUTPUT_DIR = Path("data_with_splits") # where the finished split will be written
LABEL_DIR  = Path("data/labels")      # folder containing your .txt label files

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

SEED        = 42    # change for a different random split; keep fixed for reproducibility
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test ratio is whatever remains (0.15)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASS MAPPING
#
# Keys   = original class index as exported by Label Studio (0-based)
# Values = new class index in your final dataset  (-1 = drop this class)
#
# HOW THE LABEL STUDIO INDICES WORK:
#   The index order comes from the PolygonLabels definition in the
#   Label Studio labeling interface (NOT from classes.txt, which was wrong).
#   The order matches the keyboard shortcuts shown in the UI:
#   rice=1, chicken=2, fish_salmon=3, ... cookie=y
#   Counting from 0, that gives the mapping below.
#
# Original Label Studio PolygonLabels order (0-indexed):
#  0  rice
#  1  chicken
#  2  fish_salmon
#  3  broccoli
#  4  carrots
#  5  salad_main           → RENAME to main_salad
#  6  wrap_half_1
#  7  wrap_half_2
#  8  pasta_pesto
#  9  bread_roll
#  10 side_salad
#  11 brownie              → MERGE into chocolate_cake (new index 6)
#  12 chocolate_cake
#  13 vanilla_pudding_with_fruits
#  14 fruit_salad
#  15 water_bottle         → RENAME to water
#  16 coffee_cup           → RENAME to coffee
#  17 tea_cup              → RENAME to tea
#  18 orange_juice_bottle  → RENAME to orange_juice
#  19 cola_can             → RENAME to cola
#  20 honey
#  21 plum_jam             (appears on every bread_roll tray — expected)
#  22 cherry_jam
#  23 butter
#  24 cookie               (appears on every tray)
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: pasta_pesto = 0 samples


OLD_TO_NEW = {
    11: 0,   # bread_roll
    12: 1,   # broccoli
    13: 6,   # chocolate_cake
    14: 2,   # butter
    15: 3,   # carrots
    16: 4,   # cherry_jam
    17: 5,   # chicken

    19: 7,   # coffee

    21: 9,   # cookie
    22: 10,  # fish_salmon
    23: 11,  # fruit_salad
    24: 12,  # honey
    25: 13,  # orange_juice

    27: 15,  # plum_jam
    28: 16,  # rice
    29: 17,  # main_salad
    30: 18,  # side_salad
    31: 19,  # tea
    32: 20,  # vanilla_pudding_with_fruits
    33: 21,  # water

    34: 22,  # wrap_half_1
    35: 23,  # wrap_half_2

    36: 8,   # cola
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FINAL CLASS NAMES
# Must be in new-index order (index 0 first, index 23 last)
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORE LOGIC (you don't need to edit below this line)
# ══════════════════════════════════════════════════════════════════════════════

def remap_label_file(src: Path):
    """
    Reads one YOLO .txt label file and returns remapped lines.
    - Lines with an index not in OLD_TO_NEW are dropped with a warning.
    - Bounding box / polygon coordinates are kept exactly as-is;
      only the class index at position 0 changes.
    """
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_cls = int(parts[0])

        if old_cls not in OLD_TO_NEW:
            print(f"  ⚠ Unknown class index {old_cls} in {src.name} — skipping line")
            continue

        new_cls = OLD_TO_NEW[old_cls]
        lines_out.append(f"{new_cls} {' '.join(parts[1:])}")
    return lines_out


def copy_pair(img_path: Path, split: str):
    """
    Copies one image to data_with_splits/<split>/images/
    and writes its remapped label to data_with_splits/<split>/labels/
    """
    label_path = LABEL_DIR / img_path.with_suffix(".txt").name

    img_out = OUTPUT_DIR / split / "images" / img_path.name
    lbl_out = OUTPUT_DIR / split / "labels" / label_path.name

    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, img_out)

    if label_path.exists():
        remapped = remap_label_file(label_path)
        lbl_out.write_text("\n".join(remapped), encoding="utf-8")
    else:
        print(f"  ⚠ No label file found for {img_path.name} — writing empty label")
        lbl_out.write_text("", encoding="utf-8")


def verify_mapping():
    """Sanity check: make sure CLASS_NAMES covers all new indices."""
    max_idx = max(v for v in OLD_TO_NEW.values())
    if max_idx >= len(CLASS_NAMES):
        raise ValueError(
            f"CLASS_NAMES has {len(CLASS_NAMES)} entries but mapping uses index {max_idx}. "
            f"Add {max_idx - len(CLASS_NAMES) + 1} more name(s) to CLASS_NAMES."
        )
    print(f"✓ Mapping verified — {len(CLASS_NAMES)} classes, max index {max_idx}")


def main():
    verify_mapping()

    # Collect all images
    images = [p for p in DATA_DIR.iterdir() if p.suffix in IMG_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found in {DATA_DIR.resolve()}")

    print(f"\nFound {len(images)} images in '{DATA_DIR}'")

    # Shuffle deterministically
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
        for img in imgs:
            copy_pair(img, split)
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
    for img in splits["train"]:
        lbl = OUTPUT_DIR / "train" / "labels" / img.with_suffix(".txt").name
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