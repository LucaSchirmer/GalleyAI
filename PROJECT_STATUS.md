# Project Status

This document identifies the canonical implementation of each GalleyAI
workflow and separates it from older experiments and generated artifacts.
When code or documentation disagrees with this file, treat this file as the
current source of truth and update it when the active workflow changes.

## Current implementations

### YOLOv11 segmentation

The active segmentation implementation is in `segmentation_YOLOv11/`.

- Train with `python segmentation_YOLOv11/train.py`.
- Evaluate a trained checkpoint with
  `python segmentation_YOLOv11/evaluate_test.py` after setting the checkpoint
  path in that script.
- The prepared dataset is read from `data_with_splits/`.
- Training output is written below `runs/segment/`.

### Siamese consumption model

The canonical Siamese implementation is `siamese_consumption_model/`. It
supports multiple image backbones and regression heads and replaces all older
Siamese implementations listed below.

- Train a neural regression head with
  `python siamese_consumption_model/training/train.py --backbone <name> --head <name>`.
- Train the hybrid gradient-boosting head with
  `python siamese_consumption_model/training/train_hybrid_gbr.py --backbone <name>`.
- Generate reports with the scripts under
  `siamese_consumption_model/training/`.
- See `siamese_consumption_model/README.md` for supported backbones, heads, and
  experiment commands.

The end-to-end inference simulation in `scripts/simulate_inference.py` imports
this implementation directly.

### Pixel-counting baseline

`pixel_counting_baseline/` is the active non-neural baseline. It estimates
consumption from the change in segmented food-mask area and is useful as a
comparison point for the Siamese models.

Run it with:

```bash
python pixel_counting_baseline/run_pixel_counting_baseline.py --split val
```

### Data preparation and review

The active repository-level utilities are under `scripts/`:

- `prepare_data.py` builds the YOLO train/validation/test dataset.
- `extract_consumption_index.py` extracts consumption annotations.
- `review_consumption_webui.py` reviews and corrects those annotations.
- `validate_data_consumed.py` validates the consumed-image dataset.
- `prepare_data_siamese_nn.py` builds before/after pair manifests.
- `simulate_inference.py` runs the segmentation and consumption models as one
  inference demonstration.
- `workflow_dataset_review.md` documents the dataset-review workflow.

## Deprecated implementations

The following directories are retained only for historical reference. Do not
add features or fixes to them and do not use them for new experiments.

| Directory | Status | Replacement |
| --- | --- | --- |
| `siamese_neural_network/` | Original single-backbone Siamese implementation | `siamese_consumption_model/` |
| `v2_siamese_nn/` | Intermediate two-head implementation | `siamese_consumption_model/` |
| `project/` | Earlier copy of the configurable multi-backbone implementation | `siamese_consumption_model/` |

These folders have not been deleted because they may contain useful experiment
history. Their documentation, commands, defaults, and output formats may be
stale.

## Data and generated artifacts

The following locations are inputs or generated outputs, not alternative code
implementations:

- `data/` and `data_consumed/`: source datasets and annotations.
- `data_with_splits/`: prepared YOLO splits.
- `data_pairs_with_splits/`: prepared Siamese pair manifests.
- `manifests/`: source pairing metadata.
- `mask_cache/`: cached segmentation detections used by consumption models.
- `models/`: model files kept at repository level.
- `runs/`, `mlruns/`, and `mlflow-artifacts/`: experiment outputs and tracking
  data.
- `siamese_consumption_model/runs/`: checkpoints and tracking data owned by the
  canonical Siamese implementation.

Do not treat generated outputs as reusable source code. Do not delete datasets,
checkpoints, or experiment artifacts without confirming that they are no longer
needed.

## Maintenance rule

New work should modify the current implementations named above. If an old
implementation contains functionality that is still needed, port it into the
canonical location instead of reviving the deprecated directory. Update this
document whenever a workflow is replaced or retired.
