"""Assemble RetinaNet with the SEW-ResNet backbone.

The pipeline is:
    SEWBackboneForDetection   → dict{'0','1','2'} of multi-scale features
    FeaturePyramidNetwork     → unified-channel pyramid with LastLevelP6P7
    RetinaNet (torchvision)   → classification + regression heads, NMS in eval

Anchor generation, focal loss, GIoU regression, and post-process NMS are all
handled by torchvision's built-in RetinaNet.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torchvision.models.detection.retinanet import (
    RetinaNet,
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelP6P7,
)

from cifar100_spikedetect.backbone import SEWBackboneForDetection


class SEWBackboneWithFPN(nn.Module):
    """Wrap SEW backbone + FPN, exposing `out_channels` for RetinaNet."""

    def __init__(self, num_steps=8, fpn_out_channels=256, spiking_fpn=False, K=4):
        super().__init__()
        self.body = SEWBackboneForDetection(num_steps=num_steps)
        self.spiking_fpn = spiking_fpn
        if spiking_fpn:
            from cifar100_spikedetect.spiking_fpn import SpikingFPN
            self.fpn = SpikingFPN(
                in_channels_list=[128, 256, 512],
                out_channels=fpn_out_channels,
                K=K,
            )
        else:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[128, 256, 512],
                out_channels=fpn_out_channels,
                extra_blocks=LastLevelP6P7(in_channels=512, out_channels=fpn_out_channels),
            )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        feats = self.body(x)            # OrderedDict of {'0','1','2'}
        return self.fpn(feats)


def build_retinanet(
    num_classes=21,
    num_steps=8,
    backbone_ckpt=None,
    backbone_source='cifar100',     # 'cifar100' | 'imagenet_ann' | 'none'
    trainable_backbone_layers=3,
    min_size=416,
    max_size=416,
    spiking_fpn=False,              # V6: use SpikingFPN instead of float FPN
    K=4,                            # I-LIF integer levels for spiking path
):
    """Return a ready-to-train RetinaNet built on the SEW-ResNet backbone.

    - num_classes includes background (VOC = 20 + bg, COCO = 80 + bg).
    - backbone_source: where to initialize backbone weights from.
        'cifar100'     — load the CIFAR-100 SEW classifier (expects `backbone_ckpt`).
        'imagenet_ann' — load torchvision ResNet-18 ImageNet weights (remapped).
        'none'         — random init (for debugging).
    - trainable_backbone_layers: 0 freezes all backbone blocks; 1/2/3 progressively
      unfreezes layer4 / layer3+4 / layer2+3+4. The stem stays trainable regardless.
    """
    backbone = SEWBackboneWithFPN(num_steps=num_steps, spiking_fpn=spiking_fpn, K=K)

    if backbone_source == 'cifar100' and backbone_ckpt is not None:
        ck = Path(backbone_ckpt)
        if ck.exists():
            backbone.body.load_weights(source='cifar100', path=str(ck))
        else:
            print(f"(warning) backbone ckpt not found: {ck} — skipping load")
    elif backbone_source == 'imagenet_ann':
        backbone.body.load_weights(source='imagenet_ann')
    elif backbone_source == 'none':
        print("(info) backbone initialized from random — no pretrained weights")
    else:
        print(f"(warning) unknown backbone_source={backbone_source} — random init")

    # Differential fine-tuning: freeze early layers per `trainable_backbone_layers`.
    # Block indices 0-1 = layer1, 2-3 = layer2, 4-5 = layer3, 6-7 = layer4.
    if trainable_backbone_layers < 4:
        # Always keep stem + conv_stem trainable (new weights).
        # Freeze layer1 if trainable_backbone_layers < 3
        freeze_layers = 4 - trainable_backbone_layers  # how many layer-groups to freeze (starting from layer1)
        block_cutoff = freeze_layers * 2  # first N blocks to freeze
        for i, block in enumerate(backbone.body.blocks):
            if i < block_cutoff:
                for p in block.parameters():
                    p.requires_grad = False

    # Anchor config: 5 FPN levels (P3 stride 8, P4 stride 16, P5 stride 32,
    # P6 stride 64, P7 stride 128). Use torchvision's retinanet defaults.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    num_anchors = anchor_generator.num_anchors_per_location()[0]

    # ImageNet normalization (matches how we ran CIFAR: the pretrained SEW
    # was trained with CIFAR mean/std, but RetinaNet applies its own
    # normalization internally on raw 0-1 tensors).
    head = None   # use default RetinaNetHead inside the model

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        head=head,
    )
    return model


if __name__ == "__main__":
    # M2 smoke test
    ck = PROJECT_ROOT / "checkpoints" / "cifar100_sew_T_best.pth"

    model = build_retinanet(
        num_classes=21,
        backbone_ckpt=str(ck) if ck.exists() else None,
        trainable_backbone_layers=3,
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {n_total:,}  trainable: {n_train:,}")

    # Eval-mode forward: list of images (not a batched tensor)
    model.eval()
    imgs = [torch.randn(3, 416, 416), torch.randn(3, 416, 416)]
    with torch.no_grad():
        out = model(imgs)
    print(f"eval output: {len(out)} images, keys={list(out[0].keys())}")
    for i, o in enumerate(out):
        print(f"  image {i}: boxes {tuple(o['boxes'].shape)}  "
              f"scores {tuple(o['scores'].shape)}  labels {tuple(o['labels'].shape)}")

    # Train-mode forward: list of images + targets, should return loss dict
    model.train()
    targets = [
        {'boxes': torch.tensor([[10.0, 10.0, 100.0, 100.0]]), 'labels': torch.tensor([5])},
        {'boxes': torch.tensor([[50.0, 50.0, 300.0, 300.0]]), 'labels': torch.tensor([12])},
    ]
    losses = model(imgs, targets)
    print(f"train losses: {{k: v.item() for k,v in losses.items()}}")
    print({k: round(v.item(), 4) for k, v in losses.items()})
