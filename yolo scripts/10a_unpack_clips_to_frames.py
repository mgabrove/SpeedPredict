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


def _take_n(lst: List[Path], n: int) -> Tuple[List[Path], List[Path]]:
    n = max(0, min(n, len(lst)))
    return lst[:n], lst[n:]


def split_clips_with_minimums(
    by_class: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_clips_train: int,
    min_clips_val: int,
    min_clips_test: int,
    seed: int,
):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    rng = random.Random(seed)

    split_map = {"train": [], "val": [], "test": []}
    split_stats = {}

    for cls, clips in by_class.items():
        clips = clips[:]
        rng.shuffle(clips)
        n = len(clips)

        d_train = int(round(n * train_ratio))
        d_val = int(round(n * val_ratio))
        d_test = n - d_train - d_val

        req_total = min_clips_train + min_clips_val + min_clips_test
        if n >= req_total:
            d_train = max(d_train, min_clips_train)
            d_val = max(d_val, min_clips_val)
            d_test = max(d_test, min_clips_test)

            total = d_train + d_val + d_test
            while total > n:
                candidates = [
                    ("train", d_train - min_clips_train),
                    ("val", d_val - min_clips_val),
                    ("test", d_test - min_clips_test),
                ]
                candidates.sort(key=lambda x: x[1], reverse=True)
                reduced = False
                for name, slack in candidates:
                    if slack > 0:
                        if name == "train":
                            d_train -= 1
                        elif name == "val":
                            d_val -= 1
                        else:
                            d_test -= 1
                        total -= 1
                        reduced = True
                        break
                if not reduced:
                    break

            while total < n:
                deficits = [
                    ("train", train_ratio - (d_train / max(1, n))),
                    ("val", val_ratio - (d_val / max(1, n))),
                    ("test", test_ratio - (d_test / max(1, n))),
                ]
                deficits.sort(key=lambda x: x[1], reverse=True)
                if deficits[0][0] == "train":
                    d_train += 1
                elif deficits[0][0] == "val":
                    d_val += 1
                else:
                    d_test += 1
                total += 1
        else:
            # best effort if very few clips
            d_train = d_val = d_test = 0
            order = ["train", "val", "test"]
            for i in range(n):
                name = order[i % 3]
                if name == "train":
                    d_train += 1
                elif name == "val":
                    d_val += 1
                else:
                    d_test += 1

        train_clips, rem = _take_n(clips, d_train)
        val_clips, rem = _take_n(rem, d_val)
        test_clips, rem = _take_n(rem, d_test)
        train_clips.extend(rem)

        split_map["train"].extend([(cls, p) for p in train_clips])
        split_map["val"].extend([(cls, p) for p in val_clips])
        split_map["test"].extend([(cls, p) for p in test_clips])

        split_stats[cls] = {
            "total_clips": n,
            "train_clips": len(train_clips),
            "val_clips": len(val_clips),
            "test_clips": len(test_clips),
        }

    return split_map, split_stats


def ensure_dirs(root: Path, include_unknown: bool):
    classes = VALID_CLASSES + (["unknown"] if include_unknown else [])
    for split in ["train", "val", "test"]:
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def count_total_frames_in_clips(clips: List[Path]) -> int:
    total = 0
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        if cap.isOpened():
            total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return total


def parse_target_map(s: str, include_unknown: bool) -> Dict[str, int]:
    out = {}
    base = VALID_CLASSES + (["unknown"] if include_unknown else [])
    if not s.strip():
        for c in base:
            out[c] = 3000
        return out

    parts = [x.strip() for x in s.split(",") if x.strip()]
    for p in parts:
        k, v = p.split(":")
        out[k.strip()] = int(v.strip())

    for c in base:
        out.setdefault(c, 3000)
    return out


def compute_auto_targets(
    available_by_split_class: Dict[str, Dict[str, int]],
    classes: List[str],
    alpha_train: float,
    alpha_val: float,
    alpha_test: float,
    max_target_per_class: int,
    min_target_per_class: int,
):
    targets = {"train": {}, "val": {}, "test": {}}
    alphas = {"train": alpha_train, "val": alpha_val, "test": alpha_test}

    for split in ["train", "val", "test"]:
        avail = available_by_split_class[split]
        positives = [avail.get(c, 0) for c in classes if avail.get(c, 0) > 0]

        if not positives:
            for c in classes:
                targets[split][c] = 0
            continue

        a_min = min(positives)
        base_target = int(round(alphas[split] * a_min))

        if max_target_per_class > 0:
            base_target = min(base_target, max_target_per_class)
        if min_target_per_class > 0:
            base_target = max(base_target, min_target_per_class)

        for c in classes:
            # cannot exceed available
            targets[split][c] = min(base_target, avail.get(c, 0))

    return targets


def compute_adaptive_stride(total_frames: int, target_frames: int, min_stride: int, max_stride: int, fallback_stride: int) -> int:
    if target_frames <= 0:
        return fallback_stride
    if total_frames <= 0:
        return fallback_stride
    raw = int(round(total_frames / float(target_frames)))
    return max(min_stride, min(max_stride, raw))


def sample_frame_indices(total_frames: int, stride: int, max_frames_per_clip: int) -> List[int]:
    idxs = list(range(0, total_frames, max(1, stride)))
    if max_frames_per_clip > 0 and len(idxs) > max_frames_per_clip:
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
    remaining_budget: int
):
    if remaining_budget <= 0:
        return {"ok": True, "saved": 0, "total_frames": 0, "skipped_budget": True}

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot_open", "saved": 0, "total_frames": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sample_frame_indices(total_frames, stride, max_frames_per_clip)

    if len(wanted) > remaining_budget:
        step = len(wanted) / float(remaining_budget)
        wanted = [wanted[int(i * step)] for i in range(remaining_budget)]

    wanted_set = set(wanted)
    saved = 0
    frame_i = -1

    stem = clip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i not in wanted_set:
            continue

        if resize_w > 0 and resize_h > 0:
            frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

        out_name = f"{stem}_f{frame_i:06d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        saved += 1

    cap.release()
    return {"ok": True, "saved": saved, "total_frames": total_frames, "skipped_budget": False}


def main():
    ap = argparse.ArgumentParser("Balanced unpack with automatic target computation")
    ap.add_argument("--clips-root", default="dataset/CLIPS")
    ap.add_argument("--out-root", default="dataset/ENV_FRAMES")
    ap.add_argument("--include-unknown", action="store_true")

    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)

    ap.add_argument("--min-clips-train", type=int, default=3)
    ap.add_argument("--min-clips-val", type=int, default=1)
    ap.add_argument("--min-clips-test", type=int, default=1)

    ap.add_argument("--fallback-stride", type=int, default=10)
    ap.add_argument("--min-stride", type=int, default=2)
    ap.add_argument("--max-stride", type=int, default=24)
    ap.add_argument("--max-frames-per-clip", type=int, default=300)

    # Manual targets (used only when --auto-targets is OFF)
    ap.add_argument("--target-train", type=str, default="")
    ap.add_argument("--target-val", type=str, default="")
    ap.add_argument("--target-test", type=str, default="")

    # Auto-target mode
    ap.add_argument("--auto-targets", action="store_true", help="Compute per-split balanced targets from available frames")
    ap.add_argument("--alpha-train", type=float, default=0.80, help="train target = alpha * min_available_class_frames_in_train")
    ap.add_argument("--alpha-val", type=float, default=0.90, help="val target = alpha * min_available_class_frames_in_val")
    ap.add_argument("--alpha-test", type=float, default=0.90, help="test target = alpha * min_available_class_frames_in_test")
    ap.add_argument("--max-target-per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--min-target-per-class", type=int, default=0, help="0 = no floor (floor is clipped by availability)")

    ap.add_argument("--jpg-quality", type=int, default=92)
    ap.add_argument("--resize-w", type=int, default=0)
    ap.add_argument("--resize-h", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    out_root = Path(args.out_root)

    items = list_clips(clips_root, include_unknown=args.include_unknown)
    if not items:
        print(f"[INFO] No clips found under: {clips_root}")
        return

    by_class = grouped_by_class(items)

    split_map, split_stats = split_clips_with_minimums(
        by_class=by_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_clips_train=args.min_clips_train,
        min_clips_val=args.min_clips_val,
        min_clips_test=args.min_clips_test,
        seed=args.seed
    )

    ensure_dirs(out_root, include_unknown=args.include_unknown)

    classes = VALID_CLASSES + (["unknown"] if args.include_unknown else [])

    # Gather available frames per split/class
    available_by_split_class = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        for cls in classes:
            cls_clips = [p for c, p in split_map[split] if c == cls]
            available_by_split_class[split][cls] = count_total_frames_in_clips(cls_clips)

    # Determine targets
    if args.auto_targets:
        targets = compute_auto_targets(
            available_by_split_class=available_by_split_class,
            classes=classes,
            alpha_train=args.alpha_train,
            alpha_val=args.alpha_val,
            alpha_test=args.alpha_test,
            max_target_per_class=args.max_target_per_class,
            min_target_per_class=args.min_target_per_class
        )
        target_mode = "auto"
    else:
        targets = {
            "train": parse_target_map(args.target_train, args.include_unknown),
            "val": parse_target_map(args.target_val, args.include_unknown),
            "test": parse_target_map(args.target_test, args.include_unknown),
        }
        # clip manual targets to availability
        for split in ["train", "val", "test"]:
            for cls in classes:
                targets[split][cls] = min(targets[split][cls], available_by_split_class[split].get(cls, 0))
        target_mode = "manual"

    # Compute adaptive stride from availability and target
    split_class_stride = {"train": {}, "val": {}, "test": {}}
    split_class_budget = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for cls in classes:
            avail = available_by_split_class[split][cls]
            tgt = targets[split][cls]
            stride = compute_adaptive_stride(
                total_frames=avail,
                target_frames=tgt,
                min_stride=args.min_stride,
                max_stride=args.max_stride,
                fallback_stride=args.fallback_stride
            )
            split_class_stride[split][cls] = stride
            split_class_budget[split][cls] = int(tgt)

    manifest = {
        "clips_root": str(clips_root),
        "out_root": str(out_root),
        "params": vars(args),
        "target_mode": target_mode,
        "split_clip_stats": split_stats,
        "available_frames_per_split_class": available_by_split_class,
        "targets_per_split_class": targets,
        "adaptive_stride": split_class_stride,
        "splits": {"train": [], "val": [], "test": []},
        "counts": {"train": {}, "val": {}, "test": {}}
    }

    for split in ["train", "val", "test"]:
        for cls in classes:
            manifest["counts"][split][cls] = 0

    for split in ["train", "val", "test"]:
        split_items = split_map[split][:]
        random.Random(args.seed + {"train": 1, "val": 2, "test": 3}[split]).shuffle(split_items)

        for cls, clip_path in split_items:
            out_dir = out_root / split / cls
            stride = split_class_stride[split][cls]
            remaining = split_class_budget[split][cls]

            info = extract_clip_frames(
                clip_path=clip_path,
                out_dir=out_dir,
                stride=stride,
                max_frames_per_clip=args.max_frames_per_clip,
                jpg_quality=args.jpg_quality,
                resize_w=args.resize_w,
                resize_h=args.resize_h,
                remaining_budget=remaining
            )

            saved = int(info.get("saved", 0))
            split_class_budget[split][cls] = max(0, split_class_budget[split][cls] - saved)
            manifest["counts"][split][cls] += saved

            rec = {
                "class": cls,
                "clip": str(clip_path),
                "saved_frames": saved,
                "total_frames": int(info.get("total_frames", 0)),
                "ok": bool(info.get("ok", False)),
                "stride_used": stride,
                "remaining_budget_after": split_class_budget[split][cls]
            }
            if not info.get("ok", False):
                rec["reason"] = info.get("reason", "unknown")
            if info.get("skipped_budget", False):
                rec["skipped_budget"] = True

            manifest["splits"][split].append(rec)

    final_totals = {}
    for split in ["train", "val", "test"]:
        final_totals[split] = sum(manifest["counts"][split].values())

    manifest["final_totals"] = final_totals
    manifest["remaining_budget"] = split_class_budget

    manifest_path = out_root / "frames_manifest_balanced.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[DONE] Balanced frame extraction complete.")
    print(f"[INFO] Target mode: {target_mode}")

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} total: {final_totals[split]}")
        for cls in classes:
            print(
                f"  {cls}: saved={manifest['counts'][split][cls]} "
                f"target={targets[split][cls]} avail={available_by_split_class[split][cls]} "
                f"stride={split_class_stride[split][cls]}"
            )

    print(f"\n[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()