"""SEW-ResNet-18 backbone for detection — supports ImageNet warm-start.

Path-A extension of cifar100_detect/backbone.py. New in this version:
  - `load_imagenet_resnet18_weights()` — loads torchvision's pretrained
    ANN ResNet-18 weights and remaps them onto our spiking SEW backbone.
    Convolutional filters transfer cleanly (identical shapes); BNTT and LIF
    start from their own defaults and adapt during COCO fine-tune.
  - `load_cifar_classifier_weights()` — original CIFAR-100 path preserved.
  - `load_weights(source=..., path=...)` — single entry point with `source`
    in {'cifar100', 'imagenet_ann'}.

Rationale: SpikingJelly has pretrained SEW-ResNet-18 ImageNet weights behind
figshare but format/timestep mismatches add integration risk. ANN-ImageNet
weights transfer to SEW convs cleanly — our BNTT+LIF wrap the convs and
only the norm/neuron state needs fresh initialization (~0.3M params of
20M), which COCO fine-tuning handles within ~5 epochs. This is the common
SNN initialization strategy used in many directly-trained SNN papers.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cifar100_sewresnet.model import SEWBlock, _bntt, _lif


class SEWBackboneForDetection(nn.Module):
    """SEW-ResNet-18 stem + residual blocks returning multi-scale features.

    Output dict (torchvision BackboneWithFPN convention):
        '0' : layer2 output  (stride 8,  128 channels)
        '1' : layer3 output  (stride 16, 256 channels)
        '2' : layer4 output  (stride 32, 512 channels)

    Stem is ImageNet-style: 7x7 stride-2 conv + BNTT + LIF + maxpool(k=3, s=2).
    """

    STRIDES = {'0': 8, '1': 16, '2': 32}
    OUT_CHANNELS = {'0': 128, '1': 256, '2': 512}

    def __init__(self, num_steps=8, beta=0.95, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps

        self.conv_stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn_stem = _bntt(num_steps, 64)
        self.lif_stem = _lif(beta, spike_grad)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

        self.output_indices = (3, 5, 7)  # end of layer2/3/4
        self.out_channels_per_stage = self.OUT_CHANNELS

    def forward(self, x, record_spikes=False):
        mem_stem = self.lif_stem.init_leaky()
        block_states = [b.init_state() for b in self.blocks]
        accum = {k: None for k in ('0', '1', '2')}

        # Spike-recording buffers: accumulate spikes per (time, layer) for viz.
        # layer1 = blocks[1] end, layer2 = blocks[3], layer3 = blocks[5], layer4 = blocks[7]
        spike_records = {'stem': [], 'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        viz_indices = {1: 'layer1', 3: 'layer2', 5: 'layer3', 7: 'layer4'}

        for t in range(self.num_steps):
            c = self.bn_stem[t](self.conv_stem(x))
            s_stem, mem_stem = self.lif_stem(c, mem_stem)
            s_stem = self.stem_pool(s_stem)
            if record_spikes:
                spike_records['stem'].append(s_stem.detach())

            h = s_stem
            for i, block in enumerate(self.blocks):
                h, block_states[i] = block.step(h, t, block_states[i])
                if i in self.output_indices:
                    key = str(self.output_indices.index(i))
                    accum[key] = h if accum[key] is None else accum[key] + h
                if record_spikes and i in viz_indices:
                    spike_records[viz_indices[i]].append(h.detach())

        out = OrderedDict()
        for k in ('0', '1', '2'):
            out[k] = accum[k] / self.num_steps

        if record_spikes:
            # Stack each layer's per-step tensors: [T, B, C, H, W]
            stacked = {k: torch.stack(v, dim=0) for k, v in spike_records.items() if v}
            return out, stacked
        return out

    # ----------------------------- weight loaders ---------------------------

    def load_cifar_classifier_weights(self, ckpt_path, verbose=True):
        """Original path: load from cifar100_sew_T classifier checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict') or ckpt
        return self._load_filtered(state, tag='cifar100', verbose=verbose)

    def load_imagenet_resnet18_weights(self, verbose=True):
        """Load torchvision's pretrained ResNet-18 ImageNet weights onto our
        SEW backbone. Conv weight shapes match; BNTT/LIF init fresh.
        """
        from torchvision.models import resnet18, ResNet18_Weights
        ann = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        ann_state = ann.state_dict()

        remap = {}
        # Stem: conv1 -> conv_stem (same 7x7 stride-2 shape)
        if 'conv1.weight' in ann_state:
            remap['conv_stem.weight'] = ann_state['conv1.weight']
        # (bn_stem is BNTT — we have T copies; seed each from ann bn1)
        for t in range(self.num_steps):
            if 'bn1.weight' in ann_state:
                remap[f'bn_stem.{t}.weight'] = ann_state['bn1.weight']
                remap[f'bn_stem.{t}.bias'] = ann_state['bn1.bias']
                remap[f'bn_stem.{t}.running_mean'] = ann_state['bn1.running_mean']
                remap[f'bn_stem.{t}.running_var'] = ann_state['bn1.running_var']

        # Residual blocks: torchvision has layer1..4 each with 2 basic blocks.
        # Our `self.blocks` is a flat ModuleList indexed 0..7:
        #   blocks[0..1] = layer1, blocks[2..3] = layer2, blocks[4..5] = layer3, blocks[6..7] = layer4
        ann_layer_map = {
            0: 'layer1.0', 1: 'layer1.1',
            2: 'layer2.0', 3: 'layer2.1',
            4: 'layer3.0', 5: 'layer3.1',
            6: 'layer4.0', 7: 'layer4.1',
        }
        for block_idx, ann_key in ann_layer_map.items():
            # conv1, bn1
            if f'{ann_key}.conv1.weight' in ann_state:
                remap[f'blocks.{block_idx}.conv1.weight'] = ann_state[f'{ann_key}.conv1.weight']
            # conv2, bn2
            if f'{ann_key}.conv2.weight' in ann_state:
                remap[f'blocks.{block_idx}.conv2.weight'] = ann_state[f'{ann_key}.conv2.weight']
            # BNTT bn1/bn2: copy ann's single BN to each of our T per-step BNs
            for t in range(self.num_steps):
                for src_bn, dst_bn in [('bn1', 'bn1'), ('bn2', 'bn2')]:
                    for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                        src_key = f'{ann_key}.{src_bn}.{suffix}'
                        dst_key = f'blocks.{block_idx}.{dst_bn}.{t}.{suffix}'
                        if src_key in ann_state:
                            remap[dst_key] = ann_state[src_key]
            # Downsample (1x1 stride-2 conv + bn)
            ann_down_conv = f'{ann_key}.downsample.0.weight'
            if ann_down_conv in ann_state:
                remap[f'blocks.{block_idx}.down_conv.weight'] = ann_state[ann_down_conv]
                for t in range(self.num_steps):
                    for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                        src_key = f'{ann_key}.downsample.1.{suffix}'
                        dst_key = f'blocks.{block_idx}.down_bn.{t}.{suffix}'
                        if src_key in ann_state:
                            remap[dst_key] = ann_state[src_key]

        return self._load_filtered(remap, tag='imagenet_ann', verbose=verbose)

    def _load_filtered(self, state, tag, verbose=True):
        own_state = self.state_dict()
        own_shapes = {k: v.shape for k, v in own_state.items()}
        to_load, skipped = {}, []
        for k, v in state.items():
            if k not in own_state:
                skipped.append(k); continue
            if v.shape != own_shapes[k]:
                skipped.append(f"{k} (shape {tuple(v.shape)} vs {tuple(own_shapes[k])})")
                continue
            to_load[k] = v
        self.load_state_dict(to_load, strict=False)
        missing = [k for k in own_state if k not in to_load]
        if verbose:
            print(f"[SEWBackboneForDetection:{tag}] loaded {len(to_load)} keys, "
                  f"skipped {len(skipped)}, still-missing {len(missing)}")
            if 0 < len(missing) < 15:
                print(f"  missing: {missing}")
        return len(to_load), len(skipped)

    def load_weights(self, source, path=None, verbose=True):
        """Unified entry point. source in {'cifar100', 'imagenet_ann'}."""
        if source == 'cifar100':
            return self.load_cifar_classifier_weights(path, verbose=verbose)
        if source == 'imagenet_ann':
            return self.load_imagenet_resnet18_weights(verbose=verbose)
        raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    net = SEWBackboneForDetection(num_steps=8)
    print("Testing ImageNet ResNet-18 weight remapping...")
    net.load_weights(source='imagenet_ann')
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 416, 416))
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}  stride {net.STRIDES[k]}, chans {net.OUT_CHANNELS[k]}")
