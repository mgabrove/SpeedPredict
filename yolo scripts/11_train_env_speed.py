#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

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


def list_class_names(split_dir: Path) -> List[str]:
    if not split_dir.exists():
        return []
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])


def count_images_in_dir(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def split_class_counts(data_root: Path, split: str) -> Dict[str, int]:
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
                    f"Fix your split so every train class appears in {split}, "
                    f"or run with --allow-missing-val-classes."
                )


class RemappedImageFolder(torch.utils.data.Dataset):
    """
    Wrap ImageFolder and remap labels to a target class list.
    Useful when val/test may miss some classes physically on disk.
    """
    def __init__(self, root: Path, transform, target_classes: List[str]):
        self.ds = datasets.ImageFolder(root, transform=transform)
        self.target_classes = target_classes
        self.target_to_idx = {c: i for i, c in enumerate(target_classes)}

        # Keep only samples whose class is in target_classes
        kept_samples = []
        for path, old_idx in self.ds.samples:
            cls_name = self.ds.classes[old_idx]
            if cls_name in self.target_to_idx:
                kept_samples.append((path, self.target_to_idx[cls_name]))

        self.samples = kept_samples
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
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Always load train normally
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    train_classes = train_ds.classes

    if not allow_missing_val_classes:
        # strict mode: every train class must be present in val/test
        validate_splits_or_raise(data_root, strict_val_test=True)
        val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
        test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

        # sanity class alignment check
        if val_ds.classes != train_classes or test_ds.classes != train_classes:
            raise RuntimeError(
                "Class folder mismatch across splits.\n"
                f"train classes: {train_classes}\n"
                f"val classes:   {val_ds.classes}\n"
                f"test classes:  {test_ds.classes}\n"
                "Please make folder names consistent across train/val/test."
            )
    else:
        # permissive mode: remap val/test to train class index space
        validate_splits_or_raise(data_root, strict_val_test=False)
        val_ds = RemappedImageFolder(data_root / "val", transform=eval_tf, target_classes=train_classes)
        test_ds = RemappedImageFolder(data_root / "test", transform=eval_tf, target_classes=train_classes)

        if len(val_ds) == 0 or len(test_ds) == 0:
            raise RuntimeError("Val/Test became empty after remapping. Please fix dataset split.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro", zero_division=0)
    return acc, f1m, ys, ps


def train_one(model_name, data_root, out_dir, epochs, lr, batch_size, img_size, workers, wd, allow_missing_val_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        data_root, img_size, batch_size, workers, allow_missing_val_classes=allow_missing_val_classes
    )
    num_classes = len(train_ds.classes)

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

    # class weights from train
    train_targets = [y for _, y in train_ds.samples]
    counts = np.bincount(train_targets, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    save_dir = Path(out_dir) / model_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item()

        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(
            f"[{model_name}] Epoch {ep}/{epochs} "
            f"loss={running / max(1, len(train_loader)):.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": img_size
            }, ckpt_path)

    # Test best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=train_ds.classes,
        digits=4,
        output_dict=True,
        zero_division=0
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
        "allow_missing_val_classes": allow_missing_val_classes
    }

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] {model_name} test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}")
    print(f"[SAVED] {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dataset/ENV_FRAMES")
    ap.add_argument("--out-dir", default="runs/env_cls")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing-val-classes", action="store_true",
                    help="Permissive mode: allow val/test to miss some classes and remap to train class space.")
    args = ap.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)

    print_split_summary(data_root)

    # Strict validation unless explicitly disabled
    try:
        validate_splits_or_raise(data_root, strict_val_test=not args.allow_missing_val_classes)
    except Exception as e:
        print(f"\n[ERROR] Dataset validation failed:\n{e}\n")
        print("Tip: either fix split coverage or rerun with --allow-missing-val-classes")
        raise

    # 1) ConvNeXt-Tiny
    train_one(
        model_name="convnext_tiny",
        data_root=data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        wd=args.wd,
        allow_missing_val_classes=args.allow_missing_val_classes,
    )

    # 2) EfficientNet-B0
    train_one(
        model_name="efficientnet_b0",
        data_root=data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        wd=args.wd,
        allow_missing_val_classes=args.allow_missing_val_classes,
    )


if __name__ == "__main__":
    main()