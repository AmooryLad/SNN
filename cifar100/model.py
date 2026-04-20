import torch
import torch.nn as nn
import snntorch as snn


class SNNConvClassifier(nn.Module):
    """Conv SNN for CIFAR-100 (3x32x32 RGB).

    Phase 3 architecture — caltech-style interleaved temporal loop, no attention.
    Depth and BN mode are configurable for incremental diagnostics.

    Variants:
      depth=4, bn_mode='bntt'   -> baseline (Phase 3 start)
      + skip connections        -> SEW-ResNet-style (Phase 3 step 2)

    Architecture (depth=4, channels 64/128/256/256):
      conv1 (3->64,  3x3, s=1, p=1) -> BN -> LIF -> pool(2): 32->16
      conv2 (64->128, 3x3, p=1)     -> BN -> LIF -> pool(2): 16->8
      conv3 (128->256, 3x3, p=1)    -> BN -> LIF -> pool(2): 8->4
      conv4 (256->256, 3x3, p=1)    -> BN -> LIF -> AdaptiveAvgPool(4x4)
      dropout -> fc -> lif_out (mem readout)
    """

    def __init__(self, num_classes=100, num_steps=4, beta=0.95, spike_grad=None,
                 bn_mode='bntt', dropout=0.5):
        super().__init__()
        if bn_mode not in ('none', 'shared', 'bntt'):
            raise ValueError(f"bn_mode must be none/shared/bntt, got {bn_mode}")

        self.num_steps = num_steps
        self.bn_mode = bn_mode
        channels = [64, 128, 256, 256]
        self.channels = channels

        lif_kwargs = dict(beta=beta, spike_grad=spike_grad,
                          learn_beta=True, learn_threshold=True)

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(**lif_kwargs)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(**lif_kwargs)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(**lif_kwargs)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)
        self.lif4 = snn.Leaky(**lif_kwargs)
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))

        if bn_mode == 'shared':
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.bn2 = nn.BatchNorm2d(channels[1])
            self.bn3 = nn.BatchNorm2d(channels[2])
            self.bn4 = nn.BatchNorm2d(channels[3])
        elif bn_mode == 'bntt':
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(channels[0]) for _ in range(num_steps)])
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(channels[1]) for _ in range(num_steps)])
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(channels[2]) for _ in range(num_steps)])
            self.bn4 = nn.ModuleList([nn.BatchNorm2d(channels[3]) for _ in range(num_steps)])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1] * 4 * 4, num_classes)
        self.lif_out = snn.Leaky(**lif_kwargs)

    def _apply_bn(self, which, x, t):
        if self.bn_mode == 'none':
            return x
        module = getattr(self, which)
        if self.bn_mode == 'shared':
            return module(x)
        return module[t](x)

    def forward(self, x, record_all=False):
        """x: [B, 3, 32, 32] -> (spk_out_rec, mem_out, mean_fire_rates).

        mem_out is membrane potential at the final time step [B, num_classes].
        Using mem readout (Fang et al. ECCV 2021) avoids output-LIF saturation.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        fire_accum = [x.new_zeros(()) for _ in range(5)]

        if record_all:
            s1_rec, s2_rec, s3_rec, s4_rec = [], [], [], []

        for t in range(self.num_steps):
            c1 = self._apply_bn('bn1', self.conv1(x), t)
            s1, mem1 = self.lif1(c1, mem1)
            p1 = self.pool1(s1)

            c2 = self._apply_bn('bn2', self.conv2(p1), t)
            s2, mem2 = self.lif2(c2, mem2)
            p2 = self.pool2(s2)

            c3 = self._apply_bn('bn3', self.conv3(p2), t)
            s3, mem3 = self.lif3(c3, mem3)
            p3 = self.pool3(s3)

            c4 = self._apply_bn('bn4', self.conv4(p3), t)
            s4, mem4 = self.lif4(c4, mem4)
            p4 = self.pool4(s4)

            flat = self.dropout(p4.flatten(1))
            out = self.fc(flat)
            spk_out, mem_out = self.lif_out(out, mem_out)
            spk_out_rec.append(spk_out)

            fire_accum[0] = fire_accum[0] + s1.mean()
            fire_accum[1] = fire_accum[1] + s2.mean()
            fire_accum[2] = fire_accum[2] + s3.mean()
            fire_accum[3] = fire_accum[3] + s4.mean()
            fire_accum[4] = fire_accum[4] + spk_out.mean()

            if record_all:
                s1_rec.append(s1.detach())
                s2_rec.append(s2.detach())
                s3_rec.append(s3.detach())
                s4_rec.append(s4.detach())

        spk_out_rec = torch.stack(spk_out_rec)
        mean_fire_rates = [f / self.num_steps for f in fire_accum]

        if record_all:
            records = {
                'lif1': torch.stack(s1_rec),
                'lif2': torch.stack(s2_rec),
                'lif3': torch.stack(s3_rec),
                'lif4': torch.stack(s4_rec),
                'out':  spk_out_rec.detach(),
            }
            return spk_out_rec, mem_out, mean_fire_rates, records

        return spk_out_rec, mem_out, mean_fire_rates
