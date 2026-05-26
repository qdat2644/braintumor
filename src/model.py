from __future__ import annotations

import timm
import torch.nn as nn

import config


SUPPORTED_MODELS = {
    "efficientnet_b0",
    "densenet121",
    "resnet50",
    "convnext_tiny",
}


def build_model(num_classes: int = 4, model_name: str | None = None) -> nn.Module:
    selected_model = model_name or config.MODEL_NAME
    if selected_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name={selected_model!r}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}"
        )
    return timm.create_model(
        selected_model,
        pretrained=True,
        num_classes=num_classes,
    )
