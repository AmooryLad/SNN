import torch
import torch.nn as nn
import snntorch as snn


class SNNConvClassifier(nn.Module):
    """Convolutional SNN for N-Caltech101.

    Energy-efficiency choices:
    - Conv layers preserve spatial structure AND share weights (far fewer
      params than FC → less memory traffic per inference on neuromorphic HW).
    - learn_beta + learn_threshold let each layer find its own efficient
      dynamics rather than forcing a uniform regime.
    - Rate-coded output (accumulated spike counts) — binary activations
      throughout, compatible with event-driven inference.
    - forward() also returns per-layer mean firing rates so the training
      loop can add a sparsity penalty (low firing = low switching energy).
    """

    def __init__(self, num_classes, num_steps, beta=0.95, spike_grad=None,
                 dropout=0.5):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              learn_beta=True, learn_threshold=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              learn_beta=True, learn_threshold=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              learn_beta=True, learn_threshold=True)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                 learn_beta=True, learn_threshold=True)

    def forward(self, x, record_all=False):
        """x: [T, B, 2, H, W] — returns (spk_out_rec, mean_fire_rates[, records]).

        spk_out_rec: [T, B, num_classes]  (sum over T outside for rate logits)
        mean_fire_rates: list of 4 scalars (one per LIF layer)

        If record_all=True, also returns a dict of per-layer spike records
        for visualization: {'lif1': [T,B,C,H,W], 'lif2': ..., 'lif3': ...,
        'out': [T,B,num_classes], 'mem_out': [T,B,num_classes]}.
        """
        x = torch.log1p(x)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        fire_accum = [x.new_zeros(()) for _ in range(4)]

        if record_all:
            s1_rec, s2_rec, s3_rec, mem_out_rec = [], [], [], []

        for t in range(self.num_steps):
            c1 = self.conv1(x[t])
            s1, mem1 = self.lif1(c1, mem1)
            p1 = self.pool1(s1)

            c2 = self.conv2(p1)
            s2, mem2 = self.lif2(c2, mem2)
            p2 = self.pool2(s2)

            c3 = self.conv3(p2)
            s3, mem3 = self.lif3(c3, mem3)
            p3 = self.pool3(s3)

            flat = self.dropout(p3.flatten(1))
            out = self.fc(flat)
            spk_out, mem_out = self.lif_out(out, mem_out)

            spk_out_rec.append(spk_out)
            fire_accum[0] = fire_accum[0] + s1.mean()
            fire_accum[1] = fire_accum[1] + s2.mean()
            fire_accum[2] = fire_accum[2] + s3.mean()
            fire_accum[3] = fire_accum[3] + spk_out.mean()

            if record_all:
                s1_rec.append(s1.detach())
                s2_rec.append(s2.detach())
                s3_rec.append(s3.detach())
                mem_out_rec.append(mem_out.detach())

        spk_out_rec = torch.stack(spk_out_rec)
        mean_fire_rates = [f / self.num_steps for f in fire_accum]

        if record_all:
            records = {
                'lif1': torch.stack(s1_rec),
                'lif2': torch.stack(s2_rec),
                'lif3': torch.stack(s3_rec),
                'out':  spk_out_rec,
                'mem_out': torch.stack(mem_out_rec),
            }
            return spk_out_rec, mean_fire_rates, records

        return spk_out_rec, mean_fire_rates
