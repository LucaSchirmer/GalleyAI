# GalleyAI — Training Pipeline Documentation

This document explains the full pipeline: how raw Label Studio exports become
two trained models — a YOLO **segmentation** model (what food is on the tray,
and where) and a Siamese **consumption regressor** (how much of it got eaten).
Read this top to bottom the first time; after that, use it as a map back to
the right script when something needs changing.

## The two-model pipeline, at a glance

```
                 ┌─────────────────────┐
  raw photos  →  │ 1. YOLO SEGMENTATION │  →  "where is each food item,
  + Label Studio │    (what & where)     │      and what class is it"
  polygon labels └─────────────────────┘
                          │
                          ▼ (trained model + cached detections)
                 ┌─────────────────────┐
  before/after   │ 2. SIAMESE NETWORK   │  →  "what % of this item
  photo pairs +  │    (how much eaten)   │      got eaten"
  pct_* labels   └─────────────────────┘
```

The two models solve different problems and are trained completely
separately, but the second one *depends on* the first: it uses the trained
YOLO model's detections (cached to disk, not re-run live) to know where each
food item is in a photo, then predicts how much of it disappeared between
the "before" and "after" photo.

---

## Directory map

```
data/                          # "before" (unconsumed) — YOLO segmentation project export
  images/                        real image files (hash-prefixed names)
  labels/                        YOLO .txt polygon labels (LOCAL index per this project)
  classes.txt                    this project's local index -> class name

data_consumed/                 # "after" (consumed) — separate Label Studio project
  images/                        real image files
  labels/                        YOLO .txt polygon labels (different local index order!)
  classes.txt                    this project's local index -> class name
  consumption_index.json         reviewed pct_*/choices annotations, keyed by image stem
  project_consumption_reviewed.json  raw reviewed export (see review UI below)

manifests/                     # raw capture-session manifests, one per main-course category
  pairs_unconsumed_chicken_rice.json
  pairs_unconsumed_fish_rice.json
  pairs_unconsumed_salad_v2.json
  pairs_unconsumed_wrap.json
    each entry: {unconsumed: <path>, consumed: [<path>, ...], possibleElements: [...]}
    paths here are the ORIGINAL raw capture layout and don't exist on disk as-is —
    only the filename STEM (after stripping the hash prefix) is trustworthy.

data_with_splits/              # YOLO segmentation train/val/test (images + remapped labels)
  train/ val/ test/               images/ + labels/ + dataset.yaml

data_pairs_with_splits/        # Siamese before/after pair train/val/test
  train.json val.json test.json   flat list of {before, after, category, numbers, choices}

mask_cache/                    # cached YOLO detections, one <stem>.json per image
                                  (covers BOTH data/images and data_consumed/images)

runs/segment/<name>/           # YOLO training output (weights, plots, confusion matrix)
runs/siamese/<name>/           # Siamese training output (best.pt checkpoint)
runs/mlflow/                   # MLflow tracking store, shared by both models

scripts/
  prepare_dataset.py             build data_with_splits/ for YOLO training
  train.py                       train the YOLO segmentation model

siamese_neutral_network/
  review_consumption_webui.py    local web UI to review/edit consumption_index.json
  prepare_data_siamese_nn.py     build data_pairs_with_splits/ (before/after pairs)
  cache_segmentation_masks.py    run YOLO once, cache detections to mask_cache/
  siamese_dataset.py             PyTorch Dataset (imported by train_siamese.py)
  siamese_model.py               model definition (imported by train_siamese.py)
  train_siamese.py               train the Siamese consumption regressor
```

---

## Stage 1 — YOLO segmentation model

**Problem it solves:** given a tray photo, find every food item and draw its
outline (polygon segmentation), labeled with a class like `rice`, `chicken`,
`bread_roll`, etc.

### 1a. The class-mapping problem

`data/` and `data_consumed/` are two *separate* Label Studio projects. Each
project exports its own `classes.txt`, and the numeric class index inside a
project's `.txt` label files is only meaningful **relative to that project's
own `classes.txt`** (line 0 = class 0, line 1 = class 1, ...). The two
projects don't use the same order, and don't even use the same names for the
same thing (`cola` vs `cola_can`, `coffee_cup` vs `coffee`, `brownie` merged
into `chocolate_cake` in one project but not the other).

**Fix (`prepare_dataset.py`):** never trust the raw numeric index across
projects. Instead:
1. Read each source's own `classes.txt` to get *its* local `index → name`.
2. Translate `name → final name` via one shared `NAME_TO_FINAL` dict (handles
   renames and merges, e.g. `"cola_can": "cola"`, `"brownie": "chocolate_cake"`).
3. Look up `final name → final index` via `CLASS_NAMES`.

Non-object classes (`Blurry`, `Consumed`, `Not consumed`, etc. — these are
Choices-field values from the *consumption* annotations, sharing the same
Label Studio project/classes.txt) map to `None` and are silently dropped;
they never have polygon geometry so they'd never legitimately appear in a
segmentation `.txt` file anyway.

### 1b. Building the split (`prepare_dataset.py`)

`DATA_SOURCES` lists both projects (`data/`, `data_consumed/`) as one
combined pool. All images from both are shuffled together (fixed `SEED=42`)
and split 70/15/15 into `data_with_splits/{train,val,test}/`. A duplicate
filename appearing in both sources raises an error rather than silently
overwriting one copy with the other.

### 1c. Training (`train.py`)

Key choices and why:
- `epochs=300, patience=50` — let early stopping decide instead of guessing
  a fixed epoch count; important since some classes have very few samples
  and can plateau or overfit quickly.
- `mask_ratio=2` (default 4) — finer mask resolution. Several classes
  (`rice`, `carrots`, `broccoli`) appear as many small, densely-packed
  instances per tray; the default downsampling was too coarse to separate them.
- `cos_lr=True`, `cache=True` — smoother convergence and faster re-reads
  over many epochs on a small dataset.

### 1d. Reading the confusion matrix

Rows = predicted class, columns = true class, each column normalized to sum
to ~1 (fraction of that class's true instances landing in each predicted
row), plus a `background` row (missed detections) and column (false
positives). Two categories of problem to watch for:

- **Data-starved classes** (e.g. `pasta_pesto` had 0 training samples,
  `orange_juice`/`wrap_half_1`/`wrap_half_2` had almost none) — no diagonal
  value at all. This is a data problem, not a model problem.
- **Genuine confusion clusters** — e.g. `chicken`/`carrots`/`rice`/`broccoli`
  cross-contaminate each other significantly. These are exactly the
  components of the composite "chicken_rice_veg" tray dish: many small,
  adjacent, sometimes-overlapping instances per image, which is inherently
  harder for both the model and the confusion-matrix IoU matching itself.

### 1e. MLflow (YOLO)

Ultralytics auto-logs to MLflow if the `mlflow` package is installed — no
code changes needed. Tracking URI defaults to `runs/mlflow` (or
`MLFLOW_TRACKING_URI` if set). Experiment name = `project=` argument to
`model.train()`; run name = `name=` argument. View with:

```bash
mlflow server --backend-store-uri runs/mlflow
# open http://127.0.0.1:5000
```

---

## Stage 2 — Consumption annotation review

`review_consumption_webui.py` is a local web UI (no relation to training
itself) for spot-checking and hand-correcting the `pct_*` and `choices`
annotations before they're used as training labels. Reads/writes
`data_consumed/consumption_index.json` in place — the shape is
`{"source_export": ..., "indexed_by_image_stem": {stem: [record]}}`, one
record per unique image stem, with `numbers` (`{field: [value]}`) and
`choices` (`{field: [[selected values]]}`).

This file is the **ground-truth label source** for Stage 3 below — every
`pct_*` percentage the Siamese network is trained to predict comes from here.

---

## Stage 3 — Siamese consumption-percentage regressor

**Problem it solves:** given a "before" (unconsumed) and "after" (consumed)
photo of the same tray, predict what percentage of a given food item was
eaten.

### 3a. Building before/after pairs (`prepare_data_siamese_nn.py`)

Source data is the per-category `manifests/*.json` files (one "meal
instance" = one `unconsumed` photo + a list of `consumed` photos taken as
the meal was progressively eaten).

**Path resolution:** the manifest's `unconsumed`/`consumed` paths are the
*original raw capture layout* and don't exist on disk anymore, and their
filenames lack the hash prefix real files carry. So paths are never trusted
literally — only the filename's **stem** (after stripping the prefix) is
used to look the real file up in `data/images` / `data_consumed/images`.

**Timestamp uniqueness:** stems are `YYYYMMDD_HHMMSS`. The *time* portion
alone recurs day to day (meals photographed around the same clock time), but
combined with the date it's globally unique — so stem-based matching is
safe, with a defensive check that raises loudly if a collision is ever
actually detected across categories.

**Joining ground truth:** each `consumed` photo's stem is looked up directly
in `consumption_index.json` to attach its `numbers`/`choices` labels.

**Splitting — pair-level, by explicit choice:** the initial design grouped
all of one meal's `consumed` photos together (never splitting them across
train/val/test) to avoid near-duplicate-photo leakage. This was
**deliberately overridden**: each photo in a group has a genuinely different,
independently meaningful consumption percentage, so pairs are now split
individually rather than by whole meal-instance group. Trade-off to keep in
mind: the tray/background can still repeat between train and test (same
meal, different eating stage), so val/test metrics likely read a bit
optimistic versus a truly unseen meal — acceptable given the priority was
having enough usable samples per split with a still-small dataset.

### 3b. Caching YOLO detections (`cache_segmentation_masks.py`)

Run once, offline, over every image in both `data/images` and
`data_consumed/images`. Output: one `mask_cache/<stem>.json` per image,
listing every detection's class, polygon (normalized 0–1), and bbox. This
exists purely so the Siamese training loop never has to re-run YOLO
inference on the same image across dozens/hundreds of epochs.

### 3c. Dataset (`siamese_dataset.py`)

Each `(before, after)` pair from `data_pairs_with_splits/*.json` expands
into **one training sample per `pct_*` field present** on that pair (a pair
with both `pct_rice` and `pct_chicken_rice_veg` becomes 2 samples).

- **Composite classes**: `pct_chicken_rice_veg`'s mask is the *union* of the
  `chicken`, `rice`, `carrots`, `broccoli` polygons — matches how that single
  percentage was actually annotated (one number for the whole "main dish"
  area, not decomposed per vegetable). This alias table
  (`METRIC_LABEL_ALIASES`) must stay in sync with the equivalent one in
  `prepare_dataset.py`.
- **Input representation**: full image (not a tight crop) + a binary mask
  channel stacked on top as a 4th channel. Chosen over cropping so the model
  keeps whole-tray context (portion size relative to the tray, etc.).
- **Missing mask in the "after" photo** (item fully eaten, YOLO detects
  nothing): mask channel is simply all zeros. No special-cased fallback
  logic — since the model always sees the full photo regardless, an empty
  mask + a visibly empty spot on the tray is itself the "fully consumed"
  signal, and the network learns this from data rather than a hand-coded rule.
- **Missing mask in the "before" photo**: dropped. There's no valid
  consumption baseline for an item that was never actually detected as
  present at the start.
- **Quality filtering**: pairs flagged `"Food rearranged significantly"` in
  `quality_flags` are dropped entirely — that flag specifically invalidates
  the "food is roughly where it was" assumption the whole approach leans on.

### 3d. Model (`siamese_model.py`)

- Shared-weight ResNet50 backbone (same weights used for both the "before"
  and "after" branch — that's what makes it a Siamese network).
- `conv1` expanded from 3 (RGB) to 4 (RGB + mask) input channels; pretrained
  RGB weights are kept, the new mask-channel weights are initialized as the
  mean of the RGB weights (a standard trick for extending a pretrained conv).
- **Backbone frozen except `conv1`.** Freezing protects against overfitting
  on a still-small dataset. `conv1` is deliberately left trainable even
  though the rest is frozen: its mask-channel weights start from a naive
  initialization (not pretrained), so if it were frozen too the network
  could never learn to actually use the mask information — that would
  quietly defeat the entire point of adding the mask channel.
- **Regression head**: concatenate `[feat_before, feat_after,
  |feat_before − feat_after|, feat_before × feat_after]` → small MLP → single
  scalar → sigmoid (0–1, matching the normalized `pct/100` target). This is
  the "Parametric" head from the project roadmap — chosen first for
  simplicity; the "Hybrid ML" (vector distance + gradient boosting) and
  "Non-Parametric" (cosine similarity) head variants are meant to be swapped
  in later without touching the backbone or dataset code.

### 3e. Training (`train_siamese.py`)

- Target normalized to `pct/100`; MSE loss; MAE is also logged in the
  original 0–100 scale for readability.
- `epochs=100, patience=15` — same early-stopping philosophy as the YOLO
  script.
- MLflow: uses the **same tracking store** as the YOLO runs
  (`runs/mlflow`), under a separate experiment name (`siamese_consumption`)
  so both models are browsable side by side in one MLflow UI. Unlike YOLO,
  this is a raw PyTorch loop, so MLflow logging is manual
  (`mlflow.log_params` once at the start, `mlflow.log_metrics` every epoch).
- Best checkpoint (by val loss) is saved to `runs/siamese/<run_name>/best.pt`
  and logged as an MLflow artifact on every improvement.

---

## Running everything, in order

```bash
# --- Stage 1: YOLO segmentation ---
python siamese_neural_network/prepare_dataset.py          # -> data_with_splits/
python siamese_neural_network/train.py                    # -> runs/segment/<name>/weights/best.pt

# --- Stage 2: review/correct consumption annotations (as needed) ---
python siamese_neural_network/review_consumption_webui.py # edits data_consumed/consumption_index.json * FLAGGING HERE NOT REALLY REQUIRED OR SHOULD BE DONE IN STAGE ONE AS THE FIRST THING 

# --- Stage 3: Siamese consumption regressor ---
python siamese_neural_network/prepare_data_siamese_nn.py  # -> data_pairs_with_splits/
python siamese_neural_network/cache_segmentation_masks.py # -> mask_cache/  (update MODEL_PATH first!)
python siamese_neural_network/train_siamese.py            # -> runs/siamese/<name>/best.pt

# --- View either model's training in MLflow ---
mlflow server --backend-store-uri runs/mlflow
# open http://127.0.0.1:5000
```

---

## Known limitations / open items (as of this writing)

- **Category coverage is lopsided**: most usable Siamese training pairs
  currently come from `chicken_rice` alone; `salad_v2`, `wrap`, and most of
  `fish_rice` show up with no linked `consumed` photos in their manifests
  despite being described as annotated — worth re-checking whether the
  manifest export step needs to be re-run for those categories.
- **`pasta_pesto`** has zero training samples in the YOLO segmentation data
  — effectively unusable until more is captured.
- **Pair-level (not group-level) splitting** means val/test metrics for the
  Siamese model should be read with the leakage caveat above in mind.
- The Siamese head is currently the simplest ("Parametric" MLP) option from
  the roadmap — the Hybrid ML and Non-Parametric variants haven't been tried
  yet.