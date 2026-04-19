import torch.nn as nn
import snntorch as snn


class SNNBackbone(nn.Module):
    """Shared SNN feature extractor — reusable across datasets."""

    def __init__(self, num_inputs, num_hidden, beta=0.95, spike_grad=None):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def init_states(self):
        return self.lif1.init_leaky(), self.lif2.init_leaky()

    def forward_step(self, x, mem1, mem2):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem1, mem2
