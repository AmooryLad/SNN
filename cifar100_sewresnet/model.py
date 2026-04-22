"""SEW-ResNet-18 for CIFAR-100 (Fang et al. NeurIPS 2021).

Key idea: in vanilla spiking-ResNet the shortcut adds membrane potentials,
which breaks the binary-spike abstraction and suffers vanishing/exploding
gradients. SEW (Spike-Element-Wise) blocks apply the LIF *inside* the residual
branch, then combine *spikes* with the shortcut via an element-wise op:

  s_out = f(s_conv, s_shortcut),   f = ADD | AND | IAND | OR

We use ADD (best on CIFAR per the paper). Output can take values in {0,1,2}
which is no longer pure binary, but stays integer and trains cleanly.

CIFAR stem: 3x3 stride-1 conv, no maxpool (keeps 32x32). Then standard
ResNet-18 layout: [2,2,2,2] blocks of (64, 128, 256, 512) channels.
"""

import torch
import torch.nn as nn
import snntorch as snn


def _bntt(num_steps, channels):
    return nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(num_steps)])


def _lif(beta, spike_grad):
    return snn.Leaky(beta=beta, spike_grad=spike_grad,
                     learn_beta=True, learn_threshold=True)


class SEWBlock(nn.Module):
    """Two-conv SEW basic block with ADD-style spike shortcut.

    main:     conv1 -> BNTT -> LIF -> conv2 -> BNTT -> LIF  = s_main
    shortcut: identity if shape matches; else conv1x1 -> BNTT -> LIF = s_short
    out = s_main + s_short   (SEW-ADD, values in {0,1,2})
    """

    def __init__(self, in_c, out_c, stride, num_steps, beta, spike_grad):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = _bntt(num_steps, out_c)
        self.lif1 = _lif(beta, spike_grad)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = _bntt(num_steps, out_c)
        self.lif2 = _lif(beta, spike_grad)

        if stride != 1 or in_c != out_c:
            self.down_conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
            self.down_bn = _bntt(num_steps, out_c)
            self.down_lif = _lif(beta, spike_grad)
        else:
            self.down_conv = None

    def init_state(self):
        st = {'m1': self.lif1.init_leaky(), 'm2': self.lif2.init_leaky()}
        if self.down_conv is not None:
            st['md'] = self.down_lif.init_leaky()
        return st

    def step(self, x, t, state):
        c1 = self.bn1[t](self.conv1(x))
        s1, state['m1'] = self.lif1(c1, state['m1'])

        c2 = self.bn2[t](self.conv2(s1))
        s_main, state['m2'] = self.lif2(c2, state['m2'])

        if self.down_conv is not None:
            cd = self.down_bn[t](self.down_conv(x))
            s_short, state['md'] = self.down_lif(cd, state['md'])
        else:
            s_short = x  # identity shortcut on the input spike train

        return s_main + s_short, state


class SEWResNet18(nn.Module):
    """SEW-ResNet-18 adapted for CIFAR (32x32 input)."""

    def __init__(self, num_classes=100, num_steps=8, beta=0.95,
                 spike_grad=None, dropout=0.0):
        super().__init__()
        self.num_steps = num_steps

        # Stem: 3x3 stride 1, keep 32x32
        self.conv_stem = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn_stem = _bntt(num_steps, 64)
        self.lif_stem = _lif(beta, spike_grad)

        # ResNet-18 layout
        specs = [
            # (in, out, stride) per block within stage
            [(64,  64, 1), (64,  64, 1)],     # layer1: 32x32
            [(64,  128, 2), (128, 128, 1)],   # layer2: 16x16
            [(128, 256, 2), (256, 256, 1)],   # layer3: 8x8
            [(256, 512, 2), (512, 512, 1)],   # layer4: 4x4
        ]

        self.blocks = nn.ModuleList()
        for stage in specs:
            for in_c, out_c, s in stage:
                self.blocks.append(
                    SEWBlock(in_c, out_c, s, num_steps, beta, spike_grad)
                )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512, num_classes)
        self.lif_out = _lif(beta, spike_grad)

    def forward(self, x):
        """x: [B, 3, 32, 32] -> (spk_out_rec, mem_out, fire_rates, logit_rec).

        fire_rates tracked per stage: [stem, layer1, layer2, layer3, layer4, out].
        logit_rec is pre-LIF fc output per time step for TET loss.
        """
        mem_stem = self.lif_stem.init_leaky()
        block_states = [b.init_state() for b in self.blocks]
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        logit_rec = []
        fire_accum = [x.new_zeros(()) for _ in range(6)]  # stem + 4 layers + out

        # blocks 0-1 = layer1, 2-3 = layer2, 4-5 = layer3, 6-7 = layer4
        layer_of = [0, 0, 1, 1, 2, 2, 3, 3]

        for t in range(self.num_steps):
            c = self.bn_stem[t](self.conv_stem(x))
            s_stem, mem_stem = self.lif_stem(c, mem_stem)
            fire_accum[0] = fire_accum[0] + s_stem.mean()

            h = s_stem
            layer_accum = [x.new_zeros(()) for _ in range(4)]
            layer_count = [0, 0, 0, 0]
            for i, block in enumerate(self.blocks):
                h, block_states[i] = block.step(h, t, block_states[i])
                li = layer_of[i]
                layer_accum[li] = layer_accum[li] + h.mean()
                layer_count[li] += 1
            for li in range(4):
                fire_accum[1 + li] = fire_accum[1 + li] + layer_accum[li] / layer_count[li]

            pooled = self.gap(h).flatten(1)
            pooled = self.dropout(pooled)
            out = self.fc(pooled)
            spk, mem_out = self.lif_out(out, mem_out)

            spk_out_rec.append(spk)
            logit_rec.append(out)
            fire_accum[5] = fire_accum[5] + spk.mean()

        spk_out_rec = torch.stack(spk_out_rec)
        logit_rec = torch.stack(logit_rec)
        mean_fire_rates = [f / self.num_steps for f in fire_accum]
        return spk_out_rec, mem_out, mean_fire_rates, logit_rec
