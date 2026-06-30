#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_class_names(split_dir: Path):
    if not split_dir.exists():
        return []
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])


def count_images_in_dir(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def split_class_counts(data_root: Path, split: str):
    split_dir = data_root / split
    out = {}
    for cls in list_class_names(split_dir):
        out[cls] = count_images_in_dir(split_dir / cls)
    return out


def print_split_summary(data_root: Path):
    print("\n[Dataset summary]")
    for split in ["train", "val", "test"]:
        counts = split_class_counts(data_root, split)
        total = sum(counts.values())
        print(f"  {split}: total={total} images, classes={len(counts)}")
        for cls, n in sorted(counts.items()):
            print(f"    - {cls}: {n}")


def validate_splits_or_raise(data_root: Path, strict_val_test: bool = True):
    train_counts = split_class_counts(data_root, "train")
    if not train_counts:
        raise RuntimeError(f"No classes/images found in {data_root / 'train'}")

    train_classes = sorted(train_counts.keys())
    missing_train = [c for c in train_classes if train_counts[c] == 0]
    if missing_train:
        raise RuntimeError(f"Train split has empty class folders: {missing_train}")

    if strict_val_test:
        for split in ["val", "test"]:
            counts = split_class_counts(data_root, split)
            if not counts:
                raise RuntimeError(f"No classes/images found in {data_root / split}")
            missing_classes = [c for c in train_classes if counts.get(c, 0) == 0]
            if missing_classes:
                raise RuntimeError(
                    f"Split '{split}' missing images for classes: {missing_classes}\n"
                    f"Fix your split or run with --allow-missing-val-classes."
                )


class RemappedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform, target_classes):
        self.ds = datasets.ImageFolder(root, transform=transform)
        self.target_classes = target_classes
        self.target_to_idx = {c: i for i, c in enumerate(target_classes)}

        kept = []
        for path, old_idx in self.ds.samples:
            cls_name = self.ds.classes[old_idx]
            if cls_name in self.target_to_idx:
                kept.append((path, self.target_to_idx[cls_name]))

        self.samples = kept
        self.classes = target_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.ds.loader(path)
        if self.ds.transform is not None:
            x = self.ds.transform(x)
        return x, y


def build_loaders(data_root, img_size, batch_size, workers, allow_missing_val_classes=False):
    data_root = Path(data_root)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),  # slightly gentler
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    train_classes = train_ds.classes

    if not allow_missing_val_classes:
        validate_splits_or_raise(data_root, strict_val_test=True)
        val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
        test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

        if val_ds.classes != train_classes or test_ds.classes != train_classes:
            raise RuntimeError(
                "Class folder mismatch across splits.\n"
                f"train={train_classes}\nval={val_ds.classes}\ntest={test_ds.classes}"
            )
    else:
        validate_splits_or_raise(data_root, strict_val_test=False)
        val_ds = RemappedImageFolder(data_root / "val", transform=eval_tf, target_classes=train_classes)
        test_ds = RemappedImageFolder(data_root / "test", transform=eval_tf, target_classes=train_classes)
        if len(val_ds) == 0 or len(test_ds) == 0:
            raise RuntimeError("Val/Test empty after remapping.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def evaluate(model, loader, device, use_amp=False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                logits = model(x)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro", zero_division=0)
    return acc, f1m, ys, ps


def save_ckpt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one(
    model_name,
    data_root,
    out_dir,
    epochs,
    lr,
    batch_size,
    img_size,
    workers,
    wd,
    allow_missing_val_classes,
    use_amp,
    grad_accum_steps,
    resume_path,
    save_every_steps,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        data_root, img_size, batch_size, workers, allow_missing_val_classes=allow_missing_val_classes
    )
    num_classes = len(train_ds.classes)

    # retry-friendly pretrained init
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

    train_targets = [y for _, y in train_ds.samples]
    counts = np.bincount(train_targets, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    save_dir = Path(out_dir) / model_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    start_epoch = 1
    best_f1 = -1.0
    global_step = 0

    # Resume
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            print(f"[{model_name}] Resuming from: {rp}")
            ckpt = torch.load(rp, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if "scaler_state" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_f1 = float(ckpt.get("best_f1", -1.0))
            global_step = int(ckpt.get("global_step", 0))
            print(f"[{model_name}] Resume epoch={start_epoch}, best_f1={best_f1:.4f}, global_step={global_step}")
        else:
            print(f"[WARN] resume path not found: {rp}. Starting fresh.")

    if grad_accum_steps < 1:
        grad_accum_steps = 1

    try:
        for ep in range(start_epoch, epochs + 1):
            model.train()
            running = 0.0
            optimizer.zero_grad(set_to_none=True)

            for i, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss = loss / grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                do_step = (i % grad_accum_steps == 0) or (i == len(train_loader))
                if do_step:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running += loss.item() * grad_accum_steps
                global_step += 1

                # periodic checkpoint (by steps)
                if save_every_steps > 0 and (global_step % save_every_steps == 0):
                    payload = {
                        "epoch": ep,
                        "global_step": global_step,
                        "best_f1": best_f1,
                        "model_name": model_name,
                        "classes": train_ds.classes,
                        "img_size": img_size,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    }
                    save_ckpt(last_path, payload)

            scheduler.step()

            val_acc, val_f1, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
            print(
                f"[{model_name}] Epoch {ep}/{epochs} "
                f"loss={running / max(1, len(train_loader)):.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

            # save last each epoch
            payload = {
                "epoch": ep,
                "global_step": global_step,
                "best_f1": best_f1,
                "model_name": model_name,
                "classes": train_ds.classes,
                "img_size": img_size,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            }
            save_ckpt(last_path, payload)

            if val_f1 > best_f1:
                best_f1 = val_f1
                payload["best_f1"] = best_f1
                save_ckpt(best_path, payload)

    except KeyboardInterrupt:
        print(f"\n[{model_name}] Interrupted by user. Saving interrupt checkpoint...")
        payload = {
            "epoch": ep if "ep" in locals() else 0,
            "global_step": global_step,
            "best_f1": best_f1,
            "model_name": model_name,
            "classes": train_ds.classes,
            "img_size": img_size,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        }
        save_ckpt(save_dir / "interrupt.pt", payload)
        print(f"[{model_name}] Saved: {save_dir / 'interrupt.pt'}")
        raise

    # test best
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device, use_amp=use_amp)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=train_ds.classes,
        digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    result = {
        "model": model_name,
        "classes": train_ds.classes,
        "best_val_f1": best_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "allow_missing_val_classes": allow_missing_val_classes,
        "amp": use_amp,
        "grad_accum_steps": grad_accum_steps,
        "batch_size": batch_size,
        "workers": workers,
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] {model_name} test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}")
    print(f"[SAVED] {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dataset/ENV_FRAMES")
    ap.add_argument("--out-dir", default="runs/env_cls")

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)   # lowered default
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)       # safer default
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)       # lowered default
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--allow-missing-val-classes", action="store_true")

    # AMP + accumulation + resume
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    ap.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume")
    ap.add_argument("--save-every-steps", type=int, default=300, help="Periodic checkpoint interval in optimizer steps (0 disables)")

    # optionally train only one model
    ap.add_argument("--model", type=str, default="both", choices=["both", "convnext_tiny", "efficientnet_b0"])

    args = ap.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)

    print_split_summary(data_root)
    validate_splits_or_raise(data_root, strict_val_test=not args.allow_missing_val_classes)

    model_list = []
    if args.model == "both":
        model_list = ["convnext_tiny", "efficientnet_b0"]
    else:
        model_list = [args.model]

    for m in model_list:
        # If training both with one --resume path, only use it when model name matches folder;
        # simplest: use resume for single-model runs.
        resume_path = args.resume
        if args.model == "both" and args.resume:
            print("[WARN] --resume with --model both can be ambiguous. Ignoring resume for safety.")
            resume_path = ""

        train_one(
            model_name=m,
            data_root=data_root,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            img_size=args.img_size,
            workers=args.workers,
            wd=args.wd,
            allow_missing_val_classes=args.allow_missing_val_classes,
            use_amp=args.amp,
            grad_accum_steps=args.grad_accum_steps,
            resume_path=resume_path,
            save_every_steps=args.save_every_steps,
        )


if __name__ == "__main__":
    main()