from pathlib import Path
import shutil

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
dataset_root = Path(r"C:\Users\Marko\Desktop\Rad AI\dataset")
label_splits = ["train", "val", "test"]  # will skip missing folders
backup_root = dataset_root / "labels_backup_before_urban_remap"

# Keep only these old IDs and remap to new contiguous IDs
# old: 1,2,3,4,5,6  => new: 0,1,2,3,4,5
id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# -------------------------------------------------
# BACKUP
# -------------------------------------------------
print(f"[INFO] Creating backup at: {backup_root}")
if backup_root.exists():
    raise FileExistsError(
        f"Backup folder already exists: {backup_root}\n"
        f"Please move/delete it first so we don't overwrite an old backup."
    )

for split in label_splits:
    src_dir = dataset_root / "labels" / split
    if not src_dir.exists():
        print(f"[WARN] Skip missing split folder: {src_dir}")
        continue

    dst_dir = backup_root / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    for txt in src_dir.rglob("*.txt"):
        rel = txt.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt, out)

print("[INFO] Backup complete.")

# -------------------------------------------------
# REMAP
# -------------------------------------------------
total_files = 0
changed_files = 0
dropped_objects = 0
kept_objects = 0
malformed_lines = 0

for split in label_splits:
    lbl_dir = dataset_root / "labels" / split
    if not lbl_dir.exists():
        continue

    for txt in lbl_dir.rglob("*.txt"):
        total_files += 1
        original = txt.read_text(encoding="utf-8", errors="ignore").splitlines()

        new_lines = []
        file_changed = False

        for ln in original:
            s = ln.strip()
            if not s:
                continue

            parts = s.split()
            if len(parts) != 5:
                malformed_lines += 1
                # keep malformed line as-is (safer), but flag it
                new_lines.append(s)
                continue

            cls_str, x, y, w, h = parts
            try:
                old_id = int(cls_str)
                # Validate numeric coords
                float(x); float(y); float(w); float(h)
            except ValueError:
                malformed_lines += 1
                new_lines.append(s)
                continue

            if old_id in id_map:
                new_id = id_map[old_id]
                if new_id != old_id:
                    file_changed = True
                new_lines.append(f"{new_id} {x} {y} {w} {h}")
                kept_objects += 1
            else:
                # Drop classes outside urban subset
                dropped_objects += 1
                file_changed = True

        # Write back (preserve empty file if nothing left)
        out_text = ("\n".join(new_lines) + "\n") if new_lines else ""
        old_text = txt.read_text(encoding="utf-8", errors="ignore")
        if out_text != old_text:
            txt.write_text(out_text, encoding="utf-8")
            changed_files += 1

print("\n[DONE] Urban remap finished.")
print(f"Total label files scanned : {total_files}")
print(f"Files changed             : {changed_files}")
print(f"Objects kept              : {kept_objects}")
print(f"Objects dropped           : {dropped_objects}")
print(f"Malformed lines found     : {malformed_lines}")
print(f"Backup location           : {backup_root}")

input()