from ultralytics import YOLO

def main():
    # 1. Load the pretrained YOLO11 Medium Segmentation weights
    model = YOLO("yolo11m-seg.pt")

    # 2. Full training run (no longer just a pipeline smoke test)
    results = model.train(
        data="./data_with_splits/dataset.yaml",  # Path to the YAML file you made

        epochs=300,                  # Upper bound — patience below will stop earlier
        patience=50,                 # Stop if val fitness hasn't improved in 50 epochs
        imgsz=640,                   # Keeps VRAM usage low; try 960 later if small
                                      # instances (rice/carrots/broccoli) still confuse
                                      # each other after this run
        batch=8,                     # Small batch size to prevent memory crashes
        device=0,                    # Uses your Nvidia GPU; set to "cpu" if you don't have one
        amp=True,                    # Enables Automatic Mixed Precision to save VRAM
        workers=2,                   # Keeps CPU data loading stable on Windows

        mask_ratio=2,                # Finer mask resolution (default 4) — helps with
                                      # the many small/overlapping rice/veg instances
        cache=True,                  # Cache images in RAM; dataset is small enough
                                      # and this gets re-read every epoch for 300 epochs
        cos_lr=True,                 # Cosine LR schedule, tends to converge more
                                      # smoothly over longer runs than linear decay
        seed=42,                     # Reproducible runs, matches the split's SEED

        project="runs/segment",      # Also becomes the MLflow experiment name
        name="baseline_v2",          # Also becomes the MLflow run name
    )

if __name__ == "__main__":
    main()