import torch.nn as nn
import snntorch as snn


class ClassificationHead(nn.Module):
    """Output head for classification tasks (swap per-dataset by num_classes)."""

    def __init__(self, num_hidden, num_classes, beta=0.95, spike_grad=None):
        super().__init__()
        self.fc = nn.Linear(num_hidden, num_classes)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def init_state(self):
        return self.lif.init_leaky()

    def forward_step(self, x, mem):
        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem
