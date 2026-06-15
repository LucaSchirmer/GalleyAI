import os

# Target the exact locations where YOLO generates these files
cache_paths = [
    "./data/labels.cache",
    "./data/labels/train.cache",
    "./data/labels/val.cache",
    "./data/labels/test.cache",
    "./data_with_splits/labels.cache",
    "./data_with_splits/labels/train.cache",
    "./data_with_splits/labels/val.cache",
    "./data_with_splits/labels/test.cache"
]

for path in cache_paths:
    if os.path.exists(path):
        try:
            os.remove(path)
            print(path, "successfully removed.")
        except OSError as e:
            print("Error removing", path, ":", e.strerror)
    else:
        print("No cache found at:", path)