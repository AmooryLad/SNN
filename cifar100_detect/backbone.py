"""SEW-ResNet-18 backbone adapted for object detection.

Reuses SEWBlock + BNTT/LIF helpers from cifar100_sewresnet/model.py. The forward
pass produces an OrderedDict of time-averaged feature maps at strides 8/16/32
(named '0', '1', '2' per torchvision's BackboneWithFPN convention).

BatchNorm2d operates on channel dim only, so BNTT trained at 32x32 works at
416x416 (or any other resolution) without modification.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the project root is importable when this module is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cifar100_sewresnet.model import SEWBlock, _bntt, _lif


class SEWBackboneForDetection(nn.Module):
    """Wraps SEW-ResNet-18 stem + residual blocks to return multi-scale features.

    Output dict (torchvision convention — keys used by BackboneWithFPN):
        '0' : layer2 output, channels=128, stride=8   (relative to 416 input)
        '1' : layer3 output, channels=256, stride=16
        '2' : layer4 output, channels=512, stride=32

    Wait — the SEW-ResNet stem has stride 1. Actual strides from 416 input:
        stem + layer1 : stride 1  → 416
        layer2        : stride 2  → 208
        layer3        : stride 2  → 104
        layer4        : stride 2  → 52

    That only gives strides of 2/4/8 (not 8/16/32). For detection we need larger
    downsampling. We insert an extra stride-2 in the stem (kernel 7, stride 2,
    padding 3) followed by a stride-2 maxpool — matching standard ResNet-18's
    ImageNet stem. This raises our total downsampling to 8/16/32.

    Keeping the original stem for classification reuse requires initializing
    the new 7x7 conv from scratch (CIFAR's 3x3 stride-1 conv weights are 3-in
    64-out and don't transfer). The layer1-4 blocks transfer cleanly.
    """

    STRIDES = {'0': 8, '1': 16, '2': 32}
    OUT_CHANNELS = {'0': 128, '1': 256, '2': 512}

    def __init__(self, num_steps=8, beta=0.95, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps

        # ImageNet-style stem: 7x7 stride-2 conv + BNTT + LIF + maxpool stride-2.
        # Total stride-4 from this stem. layer2-4 add 3 more stride-2 hops → 8/16/32.
        self.conv_stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn_stem = _bntt(num_steps, 64)
        self.lif_stem = _lif(beta, spike_grad)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks: same layout as SEWResNet18.
        specs = [
            [(64,  64,  1), (64,  64,  1)],   # layer1
            [(64,  128, 2), (128, 128, 1)],   # layer2
            [(128, 256, 2), (256, 256, 1)],   # layer3
            [(256, 512, 2), (512, 512, 1)],   # layer4
        ]
        self.blocks = nn.ModuleList()
        for stage in specs:
            for in_c, out_c, s in stage:
                self.blocks.append(SEWBlock(in_c, out_c, s, num_steps, beta, spike_grad))

        # Block indices where each layer ends (0-indexed).
        # blocks[0..1] = layer1, blocks[2..3] = layer2, blocks[4..5] = layer3, blocks[6..7] = layer4
        self.output_indices = (3, 5, 7)  # layer2_end, layer3_end, layer4_end

        # Exposed for BackboneWithFPN. torchvision expects an integer `out_channels`
        # attribute after FPN is applied; pre-FPN it reads `in_channels_list` that
        # we'll pass explicitly. But some APIs also call `self.out_channels`, so we
        # set it to the FPN's output width (256) only AFTER wrapping; here we just
        # keep the dict for debugging/inspection.
        self.out_channels_per_stage = self.OUT_CHANNELS

    def forward(self, x):
        """x: [B, 3, H, W]. Returns OrderedDict of time-averaged features."""
        mem_stem = self.lif_stem.init_leaky()
        block_states = [b.init_state() for b in self.blocks]

        # Accumulators for time-averaged features at each output stage.
        accum = {k: None for k in ('0', '1', '2')}

        for t in range(self.num_steps):
            # Stem
            c = self.bn_stem[t](self.conv_stem(x))
            s_stem, mem_stem = self.lif_stem(c, mem_stem)
            s_stem = self.stem_pool(s_stem)

            h = s_stem
            for i, block in enumerate(self.blocks):
                h, block_states[i] = block.step(h, t, block_states[i])

                # Capture output at end of layer2, layer3, layer4.
                if i in self.output_indices:
                    key = str(self.output_indices.index(i))
                    if accum[key] is None:
                        accum[key] = h
                    else:
                        accum[key] = accum[key] + h

        # Mean over timesteps, convert to plain floats (not integer spike sums).
        out = OrderedDict()
        for k in ('0', '1', '2'):
            out[k] = accum[k] / self.num_steps
        return out

    def load_classifier_weights(self, ckpt_path, strict=False, verbose=True):
        """Load weights from a cifar100_sewresnet SEWResNet18 checkpoint.

        Skips keys not present in this module (`gap`, `fc`, `lif_out`) and the
        stem (since our ImageNet-style 7x7 stem has different shapes than the
        3x3 CIFAR stem). Returns (loaded, skipped) key counts.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict') or ckpt
        if state is None:
            raise ValueError(f"Could not find state dict in {ckpt_path}")

        own_keys = set(self.state_dict().keys())
        own_shapes = {k: v.shape for k, v in self.state_dict().items()}

        to_load = {}
        skipped = []
        for k, v in state.items():
            if k not in own_keys:
                skipped.append(k)
                continue
            if v.shape != own_shapes[k]:
                skipped.append(f"{k} (shape mismatch {tuple(v.shape)} vs {tuple(own_shapes[k])})")
                continue
            to_load[k] = v

        missing = [k for k in own_keys if k not in to_load]
        res = self.load_state_dict(to_load, strict=False)
        if verbose:
            print(f"[SEWBackboneForDetection] loaded {len(to_load)} keys, "
                  f"skipped {len(skipped)}, own-keys missing {len(missing)}")
            if missing and len(missing) < 15:
                print(f"  missing: {missing}")
        return len(to_load), len(skipped)


if __name__ == "__main__":
    # Smoke test (M1)
    net = SEWBackboneForDetection(num_steps=8)
    ck = PROJECT_ROOT / "checkpoints" / "cifar100_sew_T_best.pth"
    if ck.exists():
        net.load_classifier_weights(str(ck))
    else:
        print(f"(no checkpoint at {ck}, skipping load test)")

    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 416, 416))
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}  (stride {net.STRIDES[k]}, channels {net.OUT_CHANNELS[k]})")
