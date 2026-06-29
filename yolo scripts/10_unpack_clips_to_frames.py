#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

VALID_CLASSES = ["30", "40", "50", "60", "70", "80"]


def list_clips(clips_root: Path, include_unknown: bool) -> List[Tuple[str, Path]]:
    classes = VALID_CLASSES + (["unknown"] if include_unknown else [])
    out = []
    for cls in classes:
        d = clips_root / cls
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            out.append((cls, p))
    return out


def grouped_by_class(items: List[Tuple[str, Path]]) -> Dict[str, List[Path]]:
    g: Dict[str, List[Path]] = {}
    for cls, p in items:
        g.setdefault(cls, []).append(p)
    return g


def split_clips(
    by_class: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    rng = random.Random(seed)

    split_map = {"train": [], "val": [], "test": []}  # list of (cls, clip_path)

    for cls, clips in by_class.items():
        clips = clips[:]
        rng.shuffle(clips)
        n = len(clips)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_clips = clips[:n_train]
        val_clips = clips[n_train:n_train + n_val]
        test_clips = clips[n_train + n_val:n_train + n_val + n_test]

        split_map["train"].extend([(cls, p) for p in train_clips])
        split_map["val"].extend([(cls, p) for p in val_clips])
        split_map["test"].extend([(cls, p) for p in test_clips])

    return split_map


def ensure_dirs(root: Path, include_unknown: bool):
    classes = VALID_CLASSES + (["unknown"] if include_unknown else [])
    for split in ["train", "val", "test"]:
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def sample_frame_indices(total_frames: int, stride: int, max_frames_per_clip: int) -> List[int]:
    idxs = list(range(0, total_frames, stride))
    if max_frames_per_clip > 0 and len(idxs) > max_frames_per_clip:
        # uniform downsample to max_frames_per_clip
        step = len(idxs) / max_frames_per_clip
        idxs = [idxs[int(i * step)] for i in range(max_frames_per_clip)]
    return idxs


def extract_clip_frames(
    clip_path: Path,
    out_dir: Path,
    stride: int,
    max_frames_per_clip: int,
    jpg_quality: int,
    resize_w: int,
    resize_h: int,
):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot_open", "saved": 0, "total_frames": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_frame_indices(total_frames, stride, max_frames_per_clip))
    saved = 0
    frame_i = -1

    stem = clip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i not in wanted:
            continue

        if resize_w > 0 and resize_h > 0:
            frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

        out_name = f"{stem}_f{frame_i:06d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        saved += 1

    cap.release()
    return {"ok": True, "saved": saved, "total_frames": total_frames}


def main():
    ap = argparse.ArgumentParser("Unpack speed clips into labeled frame dataset")
    ap.add_argument("--clips-root", default="dataset/CLIPS", help="Root folder containing class subfolders with mp4 clips")
    ap.add_argument("--out-root", default="dataset/ENV_FRAMES", help="Output dataset root")
    ap.add_argument("--include-unknown", action="store_true", help="Include unknown class")
    ap.add_argument("--stride", type=int, default=10, help="Keep 1 frame every N frames")
    ap.add_argument("--max-frames-per-clip", type=int, default=300, help="Cap frames extracted per clip (0 = no cap)")
    ap.add_argument("--jpg-quality", type=int, default=92)
    ap.add_argument("--resize-w", type=int, default=0, help="Optional resize width (0 = keep original)")
    ap.add_argument("--resize-h", type=int, default=0, help="Optional resize height (0 = keep original)")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    out_root = Path(args.out_root)

    items = list_clips(clips_root, include_unknown=args.include_unknown)
    if not items:
        print(f"[INFO] No clips found under: {clips_root}")
        return

    by_class = grouped_by_class(items)
    split_map = split_clips(
        by_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    ensure_dirs(out_root, include_unknown=args.include_unknown)

    manifest = {
        "clips_root": str(clips_root),
        "out_root": str(out_root),
        "params": vars(args),
        "splits": {"train": [], "val": [], "test": []},
        "counts": {"train": 0, "val": 0, "test": 0},
    }

    for split in ["train", "val", "test"]:
        for cls, clip_path in split_map[split]:
            out_dir = out_root / split / cls
            info = extract_clip_frames(
                clip_path=clip_path,
                out_dir=out_dir,
                stride=args.stride,
                max_frames_per_clip=args.max_frames_per_clip,
                jpg_quality=args.jpg_quality,
                resize_w=args.resize_w,
                resize_h=args.resize_h,
            )
            rec = {
                "class": cls,
                "clip": str(clip_path),
                "saved_frames": info["saved"],
                "total_frames": info["total_frames"],
                "ok": info["ok"],
            }
            if not info["ok"]:
                rec["reason"] = info["reason"]

            manifest["splits"][split].append(rec)
            manifest["counts"][split] += info["saved"]

    manifest_path = out_root / "frames_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[DONE] Frame extraction complete.")
    print(f"[DONE] Train frames: {manifest['counts']['train']}")
    print(f"[DONE] Val frames:   {manifest['counts']['val']}")
    print(f"[DONE] Test frames:  {manifest['counts']['test']}")
    print(f"[DONE] Manifest:     {manifest_path}")


if __name__ == "__main__":
    main()