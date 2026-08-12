# Siamese Consumption Model

## Folder structure

```
project/
├── data/
│   └── siamese_dataset.py        # ConsumptionPairDataset (unchanged from your original)
├── models/
│   ├── backbones.py               # 5 backbones, all patched for 4-channel (RGB+mask) input
│   ├── siamese_net.py             # SiameseConsumptionNet(backbone_name, head_name)
│   └── heads/
│       ├── mlp_head.py            # Parametric: Concatenation + MLP
│       ├── distance_head.py       # Non-parametric: Cosine / Euclidean
│       └── classification_head.py # Shared binary head (drinks/extras/cookie)
├── training/
│   ├── engine.py                  # run_epoch, shared by train.py
│   ├── train.py                   # single entry point: pick --backbone + --head
│   └── train_hybrid_gbr.py        # Hybrid ML head (separate, see below)
├── configs/                       # (empty -- reserved if you want YAML configs later)
├── data_pairs_with_splits/        # put train.json / val.json here
├── mask_cache/                    # put your YOLO detection caches here
├── runs/
│   ├── mlflow/                    # sqlite MLflow store lives here (mlflow.db)
│   └── siamese/<run_name>/best.pt # checkpoints, one folder per run
└── requirements.txt
```

## What's implemented

**Backbones** (`models/backbones.py`), matching your midterm slide:
| name                | source     | notes |
|---------------------|------------|-------|
| `resnet50`           | torchvision | your original |
| `convnext_tiny`      | timm       | |
| `mobilenetv3_large`  | timm       | |
| `vit_b16`            | timm       | resized to 224x224 internally (fixed pos. embeddings) |
| `swin_t`             | timm       | resized to 224x224 internally (fixed pos. embeddings) |

All five get the same 4th-channel patch treatment as your original ResNet50:
pretrained RGB weights copied over unchanged, mask channel mean-initialized,
backbone frozen except for that one patched conv (so the mask channel can
still learn even when everything else is frozen).

**Regression heads** (`models/heads/`), matching your midterm slide:
| name        | type          | what it is |
|-------------|---------------|------------|
| `mlp`       | Parametric    | `[a,b,\|a-b\|,a*b]` → MLP trunk → 1 output (your original head) |
| `cosine`    | Non-parametric| cosine similarity + learned scale/bias calibration |
| `euclidean` | Non-parametric| normalized Euclidean distance + learned scale/bias calibration |
| *(hybrid_gbr)* | Hybrid ML  | **not** in this list — see below |

The classification head (drinks/extras/cookie) is always attached regardless
of which regression head you pick — it needs its own small trunk since the
distance heads have no shared representation to reuse.

## Running experiments

Every combination of backbone × gradient-trained head goes through one script:

```bash
python training/train.py --backbone resnet50 --head mlp
python training/train.py --backbone convnext_tiny --head cosine
python training/train.py --backbone vit_b16 --head euclidean --epochs 50 --batch-size 8
python training/train.py --help   # full flag list (lr, patience, freeze, etc.)
```

The **Hybrid ML** head (embeddings → Gradient Boosting Regressor) has no
gradient-trained weights at all, so it can't share the epoch/optimizer loop —
it gets its own script, but logs to the *same* MLflow experiment so it still
shows up next to everything else for comparison:

```bash
python training/train_hybrid_gbr.py --backbone resnet50
python training/train_hybrid_gbr.py --backbone convnext_tiny --n-estimators 500
```

All runs (regression heads and the GBR script alike) log to the same MLflow
experiment `siamese_consumption`, tracked in `runs/mlflow/mlflow.db` (SQLite —
newer MLflow versions deprecated the plain file store, so this replaces the
`runs/mlflow` directory approach your original script used). View it with:

```bash
mlflow server --backend-store-uri sqlite:///runs/mlflow/mlflow.db
```

For a full sweep across everything (RQ2: "which architecture is best"):

```bash
for backbone in resnet50 convnext_tiny mobilenetv3_large vit_b16 swin_t; do
  for head in mlp cosine euclidean; do
    python training/train.py --backbone $backbone --head $head
  done
  python training/train_hybrid_gbr.py --backbone $backbone
done
```

## Notes / things to double check on your machine

- **ViT-B/16 / Swin-T VRAM**: these are heavier than the CNNs at comparable
  batch sizes — drop `--batch-size` if you hit OOM.
- **timm pretrained downloads**: first run of each timm backbone will
  download weights from Hugging Face Hub / timm's release assets — make
  sure that's reachable from wherever you run this.
- I could not fully exercise pretrained-weight downloads or a real dataset
  in this sandbox (network + disk are restricted here), but the full
  pipeline was smoke-tested end-to-end with `pretrained=False` and a
  synthetic dataset: dataset loading → all 5 backbones → all 3 heads →
  training loop → MLflow logging → checkpointing → the GBR script. Worth
  running one quick real epoch on your machine before a long sweep, just
  to confirm the pretrained-weight downloads work in your environment.
