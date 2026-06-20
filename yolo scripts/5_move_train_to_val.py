from pathlib import Path
import random
import shutil

random.seed(42)  # reproducible split

root = Path(r"C:\Users\Marko\Desktop\Rad AI\dataset")
train_img = root / "images" / "train"
val_img = root / "images" / "val"
train_lbl = root / "labels" / "train"
val_lbl = root / "labels" / "val"

val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = [p for p in train_img.iterdir() if p.suffix.lower() in image_exts]

# move 20% to val (change if you want)
n_val = max(1, int(len(images) * 0.2))
to_move = random.sample(images, n_val)

for img_path in to_move:
    # move image
    shutil.move(str(img_path), str(val_img / img_path.name))

    # move matching label if exists; else create empty in val
    lbl_name = img_path.stem + ".txt"
    src_lbl = train_lbl / lbl_name
    dst_lbl = val_lbl / lbl_name
    if src_lbl.exists():
        shutil.move(str(src_lbl), str(dst_lbl))
    else:
        dst_lbl.touch()

print(f"Moved {len(to_move)} images from train -> val")

input()