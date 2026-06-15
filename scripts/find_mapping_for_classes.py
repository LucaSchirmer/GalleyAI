import os
import cv2
import glob
import numpy as np

LABEL_DIR = "data/labels"
IMAGE_DIR = "data/images"

TARGET_INDICES = [
    11,12,13,14,15,16,
    17,18,19,20,21,22,
    23,24,25,26,27,28,
    29,30,31,32,33,34,
    35,36,37,
]

OUTPUT_FILE = "class_mapping_manual.txt"


def find_image(label_file):
    stem = os.path.splitext(os.path.basename(label_file))[0]

    matches = []

    for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
        matches.extend(
            glob.glob(
                os.path.join(
                    IMAGE_DIR,
                    "**",
                    f"{stem}.{ext}"
                ),
                recursive=True
            )
        )

    return matches[0] if matches else None


for target_idx in TARGET_INDICES:

    label_file = None

    for f in glob.glob(os.path.join(LABEL_DIR, "*.txt")):

        with open(f) as fp:
            for line in fp:

                if line.startswith(f"{target_idx} "):
                    label_file = f
                    break

        if label_file:
            break

    if label_file is None:
        print(f"Index {target_idx}: no file found")
        continue

    image_file = find_image(label_file)

    if image_file is None:
        print(f"Image missing for {label_file}")
        continue

    img = cv2.imread(image_file)

    h, w = img.shape[:2]

    with open(label_file) as fp:

        for line in fp:

            parts = line.strip().split()

            cls = int(parts[0])

            coords = np.array(
                list(map(float, parts[1:])),
                dtype=np.float32
            ).reshape(-1, 2)

            coords[:, 0] *= w
            coords[:, 1] *= h

            pts = coords.astype(np.int32)

            if cls == target_idx:
                color = (0, 0, 255)      # RED
                thickness = 4
            else:
                color = (0, 255, 0)      # GREEN
                thickness = 2

            cv2.polylines(
                img,
                [pts],
                True,
                color,
                thickness
            )

            x = pts[:, 0].min()
            y = pts[:, 1].min()

            cv2.putText(
                img,
                str(cls),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    out_file = f"/tmp/index_{target_idx}.jpg"

    cv2.imwrite(out_file, img)

    os.system(f'explorer.exe "$(wslpath -w {out_file})"')
    
    name = input(
        f"\nIndex {target_idx} = "
    ).strip()

    with open(OUTPUT_FILE, "a") as fp:
        fp.write(
            f"{target_idx},{name}\n"
        )

    cv2.destroyAllWindows()

print("\nDone.")