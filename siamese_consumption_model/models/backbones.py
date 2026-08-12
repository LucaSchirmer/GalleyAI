"""Backbone factory for the Siamese consumption model.

Every backbone here is adapted to accept a 4th input channel (RGB + binary
mask) using the SAME technique used in the original ResNet50 implementation:
copy the pretrained 3-channel conv weights over unchanged, and initialize
the new 4th (mask) channel as the mean of the pretrained RGB weights across
the input-channel dim. This keeps the backbone numerically identical to its
pretrained self on the RGB channels at initialization, so training starts
from a known-good feature extractor and only has to learn how to *use* the
mask channel -- it isn't already using it.

"Backbone frozen except conv1" reasoning from the original model carries
over to every architecture here: `freeze=True` means "freeze everything
except the patched first conv" (whatever module that is for this
architecture), since that layer's mask-channel weights aren't pretrained
and need to stay trainable for the mask to have any effect at all.

ViT-B/16 and Swin-T use learned positional embeddings tied to a fixed
224x224 input (the patch grid size depends on input resolution). Rather
than depend on per-timm-version dynamic-resolution support, this wrapper
resizes to 224x224 internally for those two backbones specifically; the
three CNN backbones (fully convolutional + global pool) accept the
dataset's native 384x384 tensors directly, no resize needed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG -- one entry per supported backbone name
# ══════════════════════════════════════════════════════════════════════════════

BACKBONE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "resnet50": dict(
        source="torchvision",
        first_conv_path="conv1",
        native_input_size=None,  # fully convolutional, accepts any size
    ),
    "convnext_tiny": dict(
        source="timm",
        timm_name="convnext_tiny",
        first_conv_path="stem.0",
        native_input_size=None,
    ),
    "mobilenetv3_large": dict(
        source="timm",
        timm_name="mobilenetv3_large_100",
        first_conv_path="conv_stem",
        native_input_size=None,
    ),
    "vit_b16": dict(
        source="timm",
        timm_name="vit_base_patch16_224",
        first_conv_path="patch_embed.proj",
        native_input_size=224,  # fixed learned positional embeddings
    ),
    "swin_t": dict(
        source="timm",
        timm_name="swin_tiny_patch4_window7_224",
        first_conv_path="patch_embed.proj",
        native_input_size=224,  # fixed learned positional embeddings
    ),
}


def list_backbones():
    return list(BACKBONE_CONFIGS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS -- generic "expand first conv to 4 channels" for any nn.Module tree
# ══════════════════════════════════════════════════════════════════════════════

def _get_parent_and_attr(model: nn.Module, dotted_path: str) -> Tuple[nn.Module, str]:
    """'stem.0' -> (model.stem, '0'); 'conv1' -> (model, 'conv1')."""
    parts = dotted_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def _get_module(model: nn.Module, dotted_path: str) -> nn.Module:
    parent, last = _get_parent_and_attr(model, dotted_path)
    return parent[int(last)] if last.isdigit() else getattr(parent, last)


def _set_module(model: nn.Module, dotted_path: str, new_module: nn.Module) -> None:
    parent, last = _get_parent_and_attr(model, dotted_path)
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def expand_first_conv_to_4ch(model: nn.Module, conv_path: str) -> nn.Conv2d:
    """Replaces the Conv2d at `conv_path` with a 4-in-channel version,
    keeping pretrained RGB weights and mean-initializing the mask channel.
    Returns the new Conv2d module so the caller can identify its parameters
    later for freeze/unfreeze bookkeeping."""
    old_conv: nn.Conv2d = _get_module(model, conv_path)
    if old_conv.in_channels == 4:
        return old_conv  # already patched

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias
    _set_module(model, conv_path, new_conv)
    return new_conv


def _infer_feature_dim(model: nn.Module, input_size: int) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 4, input_size, input_size)
        out = model(dummy)
    return out.shape[-1]


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER -- uniform interface regardless of backbone source
# ══════════════════════════════════════════════════════════════════════════════

class Backbone(nn.Module):
    """Uniform wrapper: .forward(x) with x = (B, 4, H, W) -> (B, feature_dim).
    `.feature_dim` and `.trainable_parameters()` are always available,
    regardless of which underlying architecture was chosen."""

    def __init__(self, name: str, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        if name not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone '{name}'. Options: {list_backbones()}")
        cfg = BACKBONE_CONFIGS[name]
        self.name = name
        self.native_input_size = cfg["native_input_size"]

        if cfg["source"] == "torchvision":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            model.fc = nn.Identity()
        else:
            model = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0)

        patched_conv = expand_first_conv_to_4ch(model, cfg["first_conv_path"])
        self.model = model
        self.feature_dim = _infer_feature_dim(model, self.native_input_size or 384)

        # Freeze everything except the patched conv's own parameters -- its
        # mask-channel weights aren't pretrained, so it must stay trainable
        # even when the rest of the backbone is frozen.
        patched_param_ids = {id(p) for p in patched_conv.parameters()}
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = id(p) in patched_param_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.native_input_size is not None and x.shape[-1] != self.native_input_size:
            # Preserve the binary semantics of the segmentation mask. RGB
            # benefits from bilinear interpolation, but bilinear-resizing the
            # fourth channel creates fractional mask edges that were never in
            # the annotation.
            rgb = F.interpolate(
                x[:, :3], size=(self.native_input_size, self.native_input_size),
                mode="bilinear", align_corners=False,
            )
            mask = F.interpolate(x[:, 3:], size=(self.native_input_size, self.native_input_size), mode="nearest")
            x = torch.cat([rgb, mask], dim=1)
        return self.model(x)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def build_backbone(name: str, pretrained: bool = True, freeze: bool = True) -> Backbone:
    return Backbone(name, pretrained=pretrained, freeze=freeze)
