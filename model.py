import torch
import torch.nn as nn
from torchvision import models

from dataset import EMOTION_NAMES


class ExpressionResNet(nn.Module):
    def __init__(self, num_classes=7, freeze_early=True, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)

        if freeze_early:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a valid Monkey Expression checkpoint.")
    return checkpoint


def checkpoint_metadata(checkpoint):
    num_classes = int(checkpoint.get("num_classes", 7))
    class_names = checkpoint.get("class_names")
    if not class_names:
        class_names = EMOTION_NAMES[:num_classes]
    label_mode = checkpoint.get("label_mode", "emotion" if num_classes == 7 else "expression")
    image_size = int(checkpoint.get("image_size", 224))
    return {
        "num_classes": num_classes,
        "class_names": list(class_names),
        "label_mode": label_mode,
        "image_size": image_size,
        "epoch": checkpoint.get("epoch"),
        "val_acc": checkpoint.get("val_acc"),
        "test_acc": checkpoint.get("test_acc"),
    }


def load_model_with_metadata(checkpoint_path, device="cpu"):
    checkpoint = load_checkpoint(checkpoint_path, device)
    metadata = checkpoint_metadata(checkpoint)
    model = ExpressionResNet(
        num_classes=metadata["num_classes"],
        freeze_early=False,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


def load_model(checkpoint_path, device="cpu"):
    checkpoint = load_checkpoint(checkpoint_path, device)
    num_classes = checkpoint.get("num_classes", 7)
    model = ExpressionResNet(num_classes=num_classes, freeze_early=False, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
