from pathlib import Path
import random
import shutil

random.seed(42)

project_root = Path(__file__).resolve().parent.parent
dataset = project_root / "dataset"

train_img = dataset / "images" / "train"
train_lbl = dataset / "labels" / "train"

# Backup destinations
backup_img = dataset / "images" / "train_empty_backup"
backup_lbl = dataset / "labels" / "train_empty_backup"

backup_img.mkdir(parents=True, exist_ok=True)
backup_lbl.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

empty_pairs = []
non_empty_count = 0

for img in train_img.iterdir():
    if not img.is_file() or img.suffix.lower() not in image_exts:
        continue

    lbl = train_lbl / f"{img.stem}.txt"
    if not lbl.exists():
        # if label missing, skip
        continue

    txt = lbl.read_text(encoding="utf-8", errors="ignore").strip()
    if txt == "":
        empty_pairs.append((img, lbl))
    else:
        non_empty_count += 1

empty_count = len(empty_pairs)
total = empty_count + non_empty_count

print(f"Non-empty images: {non_empty_count}")
print(f"Empty images:     {empty_count}")
print(f"Total:            {total}")

# Choose target empty ratio in TRAIN after moving (0.4 to 0.5 is good)
target_empty_ratio = 0.30

# E/(E+P)=r  =>  E = r*P/(1-r)
desired_empty = int((target_empty_ratio * non_empty_count) / (1 - target_empty_ratio))
to_move = max(0, empty_count - desired_empty)

print(f"Target empty ratio: {target_empty_ratio:.2f}")
print(f"Desired empty count: {desired_empty}")
print(f"Will move to backup: {to_move}")

if to_move > 0:
    selected = random.sample(empty_pairs, min(to_move, empty_count))

    for img, lbl in selected:
        dst_img = backup_img / img.name
        dst_lbl = backup_lbl / lbl.name

        # If collision, append suffix
        n = 1
        while dst_img.exists():
            dst_img = backup_img / f"{img.stem}__{n}{img.suffix}"
            n += 1

        n = 1
        while dst_lbl.exists():
            dst_lbl = backup_lbl / f"{lbl.stem}__{n}{lbl.suffix}"
            n += 1

        shutil.move(str(img), str(dst_img))
        shutil.move(str(lbl), str(dst_lbl))

print("Done. Empty samples moved to backup.")
print(f"Backup image folder: {backup_img}")
print(f"Backup label folder: {backup_lbl}")

input()