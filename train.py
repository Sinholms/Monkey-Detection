import os
import time
import argparse
import csv
import json
import random

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import (
    FER2013Dataset,
    VALID_LABEL_MODES,
    get_class_names,
    get_train_transforms,
    get_val_transforms,
    compute_class_weights,
)
from model import ExpressionResNet


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def describe_device(device):
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        print(f"Device: cuda ({name}, {total_gb:.1f} GiB VRAM)")
    else:
        print("Device: cpu")
        print("WARNING: CUDA is not available. Training will be much slower on CPU.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def create_plateau_scheduler(optimizer):
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=True
        )
    except TypeError:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label="Train Accuracy")
    ax2.plot(val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(preds, labels, class_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_metrics_csv(rows, save_path):
    if not rows:
        return
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Epoch metrics saved to {save_path}")


def save_classification_report(labels, preds, class_names, save_path):
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")


def save_config(args, class_names, save_path):
    config = vars(args).copy()
    config["class_names"] = class_names
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Expression ResNet50 on FER2013")
    parser.add_argument("--data", type=str, default="data/fer2013.csv",
                        help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-mode", choices=sorted(VALID_LABEL_MODES), default="emotion",
                        help="emotion (default) trains all 7 FER2013 emotions; expression trains 3 app targets.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Training device. Use --device cuda to fail fast if CUDA is unavailable.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="CrossEntropy label smoothing.")
    parser.add_argument("--early-stopping-patience", type=int, default=12,
                        help="Stop after this many epochs without validation improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=0.001,
                        help="Minimum validation accuracy improvement required to reset early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training on CUDA.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Initialize ResNet50 without ImageNet weights.")
    parser.add_argument("--ferplus", type=str, default=None,
                        help="Path to fer2013new.csv for FERPlus cleaner labels.")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    class_names = get_class_names(args.label_mode)
    save_config(args, class_names, os.path.join(args.save_dir, "training_config.json"))

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print("\nTo download FER2013:")
        print("  1. Install kaggle CLI: pip install kaggle")
        print("  2. Place kaggle.json in ~/.kaggle/")
        print("  3. Run: kaggle datasets download -d msambare/fer2013")
        print("  4. Extract to data/fer2013.csv")
        return

    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return
    describe_device(device)

    try:
        ds_kwargs = dict(label_mode=args.label_mode)
        if args.ferplus:
            if not os.path.exists(args.ferplus):
                print(f"ERROR: FERPlus CSV not found at {args.ferplus}")
                return
            ds_kwargs["ferplus_csv"] = args.ferplus
            print(f"Using FERPlus labels from {args.ferplus}")
        train_dataset = FER2013Dataset(args.data, split="Training",
                                       transform=get_train_transforms(args.image_size),
                                       **ds_kwargs)
        val_dataset = FER2013Dataset(args.data, split="PublicTest",
                                     transform=get_val_transforms(args.image_size),
                                     **ds_kwargs)
        test_dataset = FER2013Dataset(args.data, split="PrivateTest",
                                      transform=get_val_transforms(args.image_size),
                                      **ds_kwargs)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=pin_memory)

    model = ExpressionResNet(
        num_classes=len(class_names),
        freeze_early=True,
        pretrained=not args.no_pretrained,
    ).to(device)

    try:
        cw_kwargs = dict(label_mode=args.label_mode)
        if args.ferplus:
            cw_kwargs["ferplus_csv"] = args.ferplus
        class_weights = compute_class_weights(
            args.data,
            split="Training",
            **cw_kwargs,
        ).to(device)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    params = [
        {"params": model.model.layer4.parameters(), "lr": args.lr * 0.1},
        {"params": model.model.fc.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = create_plateau_scheduler(optimizer)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_preds, best_labels = [], []
    metrics_rows = []

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        lr_layer4 = optimizer.param_groups[0]["lr"]
        lr_fc = optimizer.param_groups[1]["lr"]
        metrics_rows.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "lr_layer4": f"{lr_layer4:.8f}",
            "lr_fc": f"{lr_fc:.8f}",
            "elapsed_seconds": f"{elapsed:.2f}",
        })
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr_fc:.5f} | {elapsed:.1f}s")

        improved = val_acc > best_val_acc + args.min_delta
        if improved or best_val_acc < 0:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_preds, best_labels = val_preds, val_labels
            checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": len(class_names),
                "class_names": class_names,
                "label_mode": args.label_mode,
                "image_size": args.image_size,
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "config": vars(args),
            }, checkpoint_path)
            print(f"  -> New best model saved (val_acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch} (val_acc: {best_val_acc:.4f})"
                )
                break

    print("-" * 70)
    checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("ERROR: Training finished without producing a checkpoint.")
        return

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    save_metrics_csv(metrics_rows, os.path.join(args.save_dir, "metrics.csv"))
    save_classification_report(
        best_labels,
        best_preds,
        class_names,
        os.path.join(args.save_dir, "classification_report_validation.txt"),
    )

    # Test evaluation
    print("\nEvaluating on test set with best model...")
    model.load_state_dict(torch.load(
        checkpoint_path,
        map_location=device, weights_only=True
    )["state_dict"])
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, use_amp=use_amp
    )
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint["test_acc"] = test_acc
    checkpoint["test_loss"] = test_loss
    torch.save(checkpoint, checkpoint_path)
    save_classification_report(
        test_labels,
        test_preds,
        class_names,
        os.path.join(args.save_dir, "classification_report_test.txt"),
    )

    # Save plots
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(args.save_dir, "training_curves.png"))
    plot_confusion_matrix(best_preds, best_labels,
                          class_names,
                          os.path.join(args.save_dir, "confusion_matrix.png"),
                          "Confusion Matrix (Validation Set)")
    plot_confusion_matrix(test_preds, test_labels,
                          class_names,
                          os.path.join(args.save_dir, "confusion_matrix_test.png"),
                          "Confusion Matrix (Test Set)")


if __name__ == "__main__":
    main()
