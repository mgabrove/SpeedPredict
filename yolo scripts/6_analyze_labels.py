from pathlib import Path
from collections import Counter, defaultdict
import math

# ===== CONFIG =====
labels_dir = Path(r"C:\Users\Marko\Desktop\Rad AI\dataset\labels\train")
images_dir = Path(r"C:\Users\Marko\Desktop\Rad AI\dataset\images\train")  # used for missing-label check

class_names = ["30", "40", "50", "60", "70", "80"]
num_classes = len(class_names)
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ===== STATS =====
class_counts = Counter()
files_with_objects = 0
empty_label_files = 0
total_label_files = 0
total_objects = 0

invalid_class_lines = []      # (file, line_no, class_id)
malformed_lines = []          # (file, line_no, line_text)
out_of_range_boxes = []       # (file, line_no, values)

box_widths = []
box_heights = []
box_areas = []

# ===== HELPERS =====
def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

# ===== CHECK LABEL FILES =====
if not labels_dir.exists():
    raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

label_files = sorted(labels_dir.rglob("*.txt"))
total_label_files = len(label_files)

for lf in label_files:
    lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
    non_empty = [ln.strip() for ln in lines if ln.strip()]

    if len(non_empty) == 0:
        empty_label_files += 1
        continue

    files_with_objects += 1

    for i, line in enumerate(non_empty, start=1):
        parts = line.split()
        if len(parts) != 5:
            malformed_lines.append((str(lf), i, line))
            continue

        c, x, y, w, h = parts
        if not (c.lstrip("-").isdigit() and is_float(x) and is_float(y) and is_float(w) and is_float(h)):
            malformed_lines.append((str(lf), i, line))
            continue

        c = int(c)
        x, y, w, h = map(float, (x, y, w, h))

        # class id range check
        if c < 0 or c >= num_classes:
            invalid_class_lines.append((str(lf), i, c))
        else:
            class_counts[c] += 1
            total_objects += 1

        # YOLO normalized box sanity
        bad = False
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            bad = True
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad = True
        # optional edge check for box boundaries
        if (x - w / 2) < 0 or (x + w / 2) > 1 or (y - h / 2) < 0 or (y + h / 2) > 1:
            bad = True

        if bad:
            out_of_range_boxes.append((str(lf), i, (x, y, w, h)))

        box_widths.append(w)
        box_heights.append(h)
        box_areas.append(w * h)

# ===== MISSING LABELS FOR IMAGES =====
missing_label_for_image = []
if images_dir.exists():
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts]
    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing_label_for_image.append(str(img))

# ===== REPORT =====
print("=" * 72)
print("YOLO LABEL ANALYSIS REPORT")
print("=" * 72)
print(f"Labels dir: {labels_dir}")
print(f"Images dir: {images_dir}")
print(f"Classes ({num_classes}): {class_names}")
print("-" * 72)
print(f"Total label files             : {total_label_files}")
print(f"Files with >=1 object         : {files_with_objects}")
print(f"Empty label files             : {empty_label_files}")
print(f"Total labeled objects         : {total_objects}")
print(f"Malformed lines               : {len(malformed_lines)}")
print(f"Invalid class-id lines        : {len(invalid_class_lines)}")
print(f"Out-of-range/invalid boxes    : {len(out_of_range_boxes)}")
print(f"Images missing label file     : {len(missing_label_for_image)}")
print("-" * 72)

print("Per-class object counts:")
for i, name in enumerate(class_names):
    print(f"  id {i:>2} ({name:>3}): {class_counts[i]}")

if total_objects > 0:
    mn_w, mx_w = min(box_widths), max(box_widths)
    mn_h, mx_h = min(box_heights), max(box_heights)
    mn_a, mx_a = min(box_areas), max(box_areas)
    avg_a = sum(box_areas) / len(box_areas)
    print("-" * 72)
    print("Box stats (normalized):")
    print(f"  width  min/max: {mn_w:.4f} / {mx_w:.4f}")
    print(f"  height min/max: {mn_h:.4f} / {mx_h:.4f}")
    print(f"  area   min/max: {mn_a:.6f} / {mx_a:.6f}   avg: {avg_a:.6f}")

# show samples of issues
def print_samples(title, items, max_n=20):
    if not items:
        return
    print("-" * 72)
    print(f"{title} (showing up to {max_n}):")
    for row in items[:max_n]:
        print(" ", row)

print_samples("Malformed lines", malformed_lines)
print_samples("Invalid class-id lines", invalid_class_lines)
print_samples("Out-of-range boxes", out_of_range_boxes)
print_samples("Images missing label file", missing_label_for_image)

print("=" * 72)

input()