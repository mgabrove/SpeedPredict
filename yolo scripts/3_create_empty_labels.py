from pathlib import Path
project_root = Path(__file__).resolve().parent.parent   # script is in "yolo scripts/"
dataset = project_root / "dataset"

# Splits you use
splits = ["train", "val", "test"]

# Image extensions to scan
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in splits:
    img_dir = dataset / "images" / split
    lbl_dir = dataset / "labels" / split

    if not img_dir.exists():
        continue

    lbl_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    total = 0

    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue

        total += 1
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            label_path.touch()  # creates empty file
            created += 1

    print(f"[{split}] images: {total}, empty labels created: {created}")
    
input()