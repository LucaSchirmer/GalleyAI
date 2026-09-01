#!/usr/bin/env python3
"""Run the complete Siamese architecture sweep for multiple random seeds.

The sweep contains 20 configurations per seed:

* 5 backbones x 3 gradient-trained heads (MLP, cosine, Euclidean)
* 5 backbones x 1 hybrid gradient-boosting regressor

Run this file from anywhere; all paths are resolved relative to the repository
root unless they are absolute. Runs execute sequentially so a single GPU is
never shared by multiple training processes.

Examples:
    python scripts/run_siamese_sweep.py --dry-run
    python scripts/run_siamese_sweep.py
    python scripts/run_siamese_sweep.py --resume
    python scripts/run_siamese_sweep.py --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "siamese_consumption_model"
TRAIN_SCRIPT = PACKAGE_ROOT / "training" / "train.py"
GBR_SCRIPT = PACKAGE_ROOT / "training" / "train_hybrid_gbr.py"
CHECKPOINT_ROOT = PACKAGE_ROOT / "runs" / "siamese"
STATUS_PATH = PACKAGE_ROOT / "runs" / "sweep_status.json"

BACKBONES = ("resnet50", "convnext_tiny", "mobilenetv3_large", "vit_b16", "swin_t")
NEURAL_HEADS = ("mlp", "cosine", "euclidean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--skip-fish-rice-veg", action="store_true")
    parser.add_argument("--train-manifest", type=Path, default=Path("data_pairs_with_splits/train.json"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data_pairs_with_splits/val.json"))
    parser.add_argument("--mask-cache-dir", type=Path, default=Path("mask_cache"))
    parser.add_argument("--resume", action="store_true", help="skip runs whose expected checkpoint already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after a failed run")
    parser.add_argument("--dry-run", action="store_true", help="print all commands without training")
    return parser.parse_args()


def resolve_from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def run_name(backbone: str, head: str, seed: int) -> str:
    return f"final_sweep_{backbone}_{head}_seed{seed}"


def expected_artifact(name: str, head: str) -> Path:
    filename = "gbr.joblib" if head == "hybrid_gbr" else "best.pt"
    return CHECKPOINT_ROOT / name / filename


def completion_marker(name: str) -> Path:
    return CHECKPOINT_ROOT / name / ".sweep_complete.json"


def build_commands(args: argparse.Namespace) -> list[tuple[str, str, int, list[str]]]:
    train_manifest = str(resolve_from_repo(args.train_manifest))
    val_manifest = str(resolve_from_repo(args.val_manifest))
    mask_cache = str(resolve_from_repo(args.mask_cache_dir))
    commands = []

    for seed in args.seeds:
        for backbone in BACKBONES:
            for head in NEURAL_HEADS:
                name = run_name(backbone, head, seed)
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--backbone", backbone,
                    "--head", head,
                    "--seed", str(seed),
                    "--run-name", name,
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--lr", str(args.lr),
                    "--backbone-lr", str(args.backbone_lr),
                    "--weight-decay", str(args.weight_decay),
                    "--regression-loss", args.regression_loss,
                    "--huber-beta", str(args.huber_beta),
                    "--train-manifest", train_manifest,
                    "--val-manifest", val_manifest,
                    "--mask-cache-dir", mask_cache,
                ]
                if args.balanced_sampling:
                    command.append("--balanced-sampling")
                if args.skip_fish_rice_veg:
                    command.append("--skip-fish-rice-veg")
                commands.append((backbone, head, seed, command))

            head = "hybrid_gbr"
            name = run_name(backbone, head, seed)
            command = [
                sys.executable,
                str(GBR_SCRIPT),
                "--backbone", backbone,
                "--seed", str(seed),
                "--run-name", name,
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--train-manifest", train_manifest,
                "--val-manifest", val_manifest,
                "--mask-cache-dir", mask_cache,
            ]
            if args.skip_fish_rice_veg:
                command.append("--skip-fish-rice-veg")
            commands.append((backbone, head, seed, command))

    return commands


def write_status(records: list[dict]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    required = (
        TRAIN_SCRIPT,
        GBR_SCRIPT,
        resolve_from_repo(args.train_manifest),
        resolve_from_repo(args.val_manifest),
        resolve_from_repo(args.mask_cache_dir),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(missing))


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must not contain duplicates")
    validate_inputs(args)
    commands = build_commands(args)
    print(f"Prepared {len(commands)} runs ({len(commands) // len(args.seeds)} configurations x {len(args.seeds)} seeds).")

    if args.dry_run:
        for index, (_, _, _, command) in enumerate(commands, start=1):
            print(f"[{index:02d}/{len(commands)}] {shlex.join(command)}")
        return 0

    records: list[dict] = []
    failures = 0
    for index, (backbone, head, seed, command) in enumerate(commands, start=1):
        name = run_name(backbone, head, seed)
        artifact = expected_artifact(name, head)
        marker = completion_marker(name)
        if args.resume and marker.exists() and artifact.exists():
            print(f"[{index:02d}/{len(commands)}] SKIP {name}: completion marker exists", flush=True)
            records.append({"run_name": name, "status": "skipped", "artifact": str(artifact)})
            write_status(records)
            continue

        print(f"[{index:02d}/{len(commands)}] START {name}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "completed" if completed.returncode == 0 else "failed"
        records.append(
            {
                "run_name": name,
                "backbone": backbone,
                "head": head,
                "seed": seed,
                "status": status,
                "return_code": completed.returncode,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(artifact),
            }
        )
        if completed.returncode == 0:
            marker.write_text(
                json.dumps({"run_name": name, "artifact": str(artifact), "finished_utc": records[-1]["finished_utc"]}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        write_status(records)
        print(f"[{index:02d}/{len(commands)}] {status.upper()} {name}", flush=True)

        if completed.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                print("Stopping after failure. Fix it, then restart with --resume.", file=sys.stderr)
                return completed.returncode

    print(f"Sweep finished: {len(commands) - failures} successful/skipped, {failures} failed.")
    print(f"Status file: {STATUS_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
