from ultralytics import YOLO

def main():
    # Point to your best trained weights
    model = YOLO("runs/segment/train-22/weights/best.pt")  # ← adjust run number

    metrics = model.val(
        data="./data_with_splits/dataset.yaml",
        split="test",        # ← this is the key, tells YOLO to use test/ instead of val/
        imgsz=640,
        device=0,
        plots=True,          # saves confusion matrix, PR curve etc.
        save_json=True,      # saves predictions as JSON if you need them later
    )

    print("\n── Test Results ──")
    print(f"  mAP50      : {metrics.seg.map50:.3f}")
    print(f"  mAP50-95   : {metrics.seg.map:.3f}")
    print(f"  Precision  : {metrics.seg.mp:.3f}")
    print(f"  Recall     : {metrics.seg.mr:.3f}")

if __name__ == "__main__":
    main()