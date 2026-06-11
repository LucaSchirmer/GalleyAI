from ultralytics import YOLO

def main():
    # 1. Load the pretrained YOLO11 Medium Segmentation weights
    model = YOLO("yolo11m-seg.pt")

    # 2. Run a lightweight baseline training session
    results = model.train(
        data="./data/dataset.yaml", # Path to the YAML file you just made
        epochs=30,                  # Low epoch count just to verify the pipeline works
        imgsz=640,                  # Keeps VRAM usage low
        batch=8,                    # Small batch size to prevent memory crashes
        device=0,                   # Uses your Nvidia GPU; set to "cpu" if you don't have one
        amp=True,                   # Enables Automatic Mixed Precision to save VRAM
        workers=2                   # Keeps CPU data loading stable on Windows
    )

if __name__ == "__main__":
    main()