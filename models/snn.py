import torch
import torch.nn as nn

from models.backbone import SNNBackbone
from models.heads import ClassificationHead


class SNNClassifier(nn.Module):
    """Full classifier: shared backbone + swappable classification head."""

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_classes,
        num_steps,
        beta=0.95,
        spike_grad=None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.backbone = SNNBackbone(num_inputs, num_hidden, beta, spike_grad=spike_grad)
        self.head = ClassificationHead(
            num_hidden, num_classes, beta, spike_grad=spike_grad
        )

    def forward(self, x, x_seq=None):
        """
        x: static input (same across time steps) — used when x_seq is None.
        x_seq: [num_steps, batch, features] for event-based data where input
               changes per time step (e.g. N-Caltech101).
        """
        mem1, mem2 = self.backbone.init_states()
        mem3 = self.head.init_state()

        spk_rec = []
        mem_rec = []

        for t in range(self.num_steps):
            x_t = x_seq[t] if x_seq is not None else x
            feat, mem1, mem2 = self.backbone.forward_step(x_t, mem1, mem2)
            spk, mem3 = self.head.forward_step(feat, mem3)
            spk_rec.append(spk)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)
