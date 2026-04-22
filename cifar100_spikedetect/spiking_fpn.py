"""Spike-native Feature Pyramid Network using I-LIF neurons.

Shape-compatible replacement for torchvision.ops.FeaturePyramidNetwork.
Same inputs/outputs (OrderedDict of multi-scale feature maps), same
`out_channels` interface. Differences:
  - Each lateral/output conv is followed by BatchNorm2d + IntegerLIF(K=4).
  - LastLevelP6P7 is also spike-native.

Conv weight shapes are identical to torchvision's FPN, so V5 checkpoints
can warm-start this module with just BN + neuron state being fresh.
"""

import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar100_spikedetect.neurons import IntegerLIF


class SpikingFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, K=4):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.lateral_bns = nn.ModuleList()
        self.lateral_lifs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.output_bns = nn.ModuleList()
        self.output_lifs = nn.ModuleList()

        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.lateral_bns.append(nn.BatchNorm2d(out_channels))
            self.lateral_lifs.append(IntegerLIF(K=K))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            self.output_bns.append(nn.BatchNorm2d(out_channels))
            self.output_lifs.append(IntegerLIF(K=K))

        # Extra blocks: spike-native LastLevelP6P7
        self.p6_conv = nn.Conv2d(in_channels_list[-1], out_channels, 3, stride=2, padding=1)
        self.p6_bn = nn.BatchNorm2d(out_channels)
        self.p6_lif = IntegerLIF(K=K)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7_bn = nn.BatchNorm2d(out_channels)
        self.p7_lif = IntegerLIF(K=K)

        self.out_channels = out_channels

    def forward(self, x):
        """x: OrderedDict with keys '0', '1', '2' (low→high stride)."""
        keys = list(x.keys())
        feats = [x[k] for k in keys]

        # Laterals: 1x1 conv + BN + LIF
        laterals = []
        for i, f in enumerate(feats):
            c = self.lateral_convs[i](f)
            c = self.lateral_bns[i](c)
            laterals.append(self.lateral_lifs[i](c))

        # Top-down merging (highest stride → lowest)
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:],
                                       mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        # Output convs + BN + LIF
        outs = []
        for i, lat in enumerate(laterals):
            o = self.output_convs[i](lat)
            o = self.output_bns[i](o)
            outs.append(self.output_lifs[i](o))

        # P6 / P7 derived from highest-stride input
        p6 = self.p6_lif(self.p6_bn(self.p6_conv(feats[-1])))
        p7 = self.p7_lif(self.p7_bn(self.p7_conv(F.relu(p6))))

        out_dict = OrderedDict()
        for i, k in enumerate(keys):
            out_dict[k] = outs[i]
        out_dict['p6'] = p6
        out_dict['p7'] = p7
        return out_dict

    def load_from_torchvision_fpn(self, tv_fpn_state_dict, verbose=True):
        """Warm-start lateral/output conv weights from a torchvision FPN.

        Torchvision FPN attribute names:
          inner_blocks.0 / .1 / .2 (1x1 lateral convs)
          layer_blocks.0 / .1 / .2 (3x3 output convs)
          extra_blocks.p6 (conv), extra_blocks.p7 (conv)
        """
        mapping = {}
        for i in range(3):
            mapping[f'inner_blocks.{i}.0.weight'] = f'lateral_convs.{i}.weight'
            mapping[f'inner_blocks.{i}.0.bias'] = f'lateral_convs.{i}.bias'
            mapping[f'layer_blocks.{i}.0.weight'] = f'output_convs.{i}.weight'
            mapping[f'layer_blocks.{i}.0.bias'] = f'output_convs.{i}.bias'
        mapping['extra_blocks.p6.weight'] = 'p6_conv.weight'
        mapping['extra_blocks.p6.bias'] = 'p6_conv.bias'
        mapping['extra_blocks.p7.weight'] = 'p7_conv.weight'
        mapping['extra_blocks.p7.bias'] = 'p7_conv.bias'

        own = self.state_dict()
        to_load = {}
        for src, dst in mapping.items():
            if src in tv_fpn_state_dict and dst in own:
                if tv_fpn_state_dict[src].shape == own[dst].shape:
                    to_load[dst] = tv_fpn_state_dict[src]
        self.load_state_dict(to_load, strict=False)
        if verbose:
            print(f"[SpikingFPN] warm-started {len(to_load)}/{len(mapping)} conv weights")
        return len(to_load)


if __name__ == "__main__":
    fpn = SpikingFPN(in_channels_list=[128, 256, 512], out_channels=256, K=4)
    x = OrderedDict()
    x['0'] = torch.randn(2, 128, 52, 52)
    x['1'] = torch.randn(2, 256, 26, 26)
    x['2'] = torch.randn(2, 512, 13, 13)
    out = fpn(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}  unique_levels={len(v.unique())}")
