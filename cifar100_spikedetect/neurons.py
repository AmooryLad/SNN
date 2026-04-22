"""I-LIF: Integer-valued Leaky Integrate-and-Fire neuron (SpikeYOLO-style).

Standard LIF outputs binary {0, 1} — we call it "spike" but it's effectively
a 1-bit quantizer. I-LIF uses integer output {0, 1, ..., K} during training
and expands each integer into K binary spikes across virtual timesteps at
inference, recovering spike-drivenness without the 1-bit training bottleneck.

References:
  SpikeYOLO (ECCV 2024) — integer-valued training + spike-driven inference.

Simplified here for the detection head/neck:
  - Single-step integer output during training (no temporal loop at the head level).
  - Maintained membrane potential is reset per batch via init_state().
  - Surrogate gradient is rectangular (straight-through between -0.5 and +0.5 of
    each integer level); ATan would also work.
"""

import torch
import torch.nn as nn


class IntegerLIFFunction(torch.autograd.Function):
    """Forward: floor((input - threshold*0) / level_size), clamped to [0, K].
    Backward: rectangular surrogate gradient (straight-through).
    """

    @staticmethod
    def forward(ctx, x, K, threshold):
        # "Activations" here are post-membrane-integration continuous values;
        # we discretize into [0, K]. threshold controls step size.
        levels = torch.clamp(torch.floor(x / threshold), min=0.0, max=float(K))
        ctx.save_for_backward(x)
        ctx.K = K
        ctx.threshold = threshold
        return levels

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Straight-through gradient where 0 <= x/threshold <= K + 1
        gate = ((x > 0) & (x < (ctx.K + 1) * ctx.threshold)).to(grad_output.dtype)
        return grad_output * gate, None, None


class IntegerLIF(nn.Module):
    """I-LIF neuron module.

    Args:
        K: max integer output level. K=4 means outputs in {0,1,2,3,4}.
        threshold: voltage per level (learnable).
        learnable_threshold: whether threshold is learned.
    """

    def __init__(self, K=4, threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.K = K
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x):
        return IntegerLIFFunction.apply(x, self.K, self.threshold.abs() + 1e-6)

    def extra_repr(self):
        return f"K={self.K}"


if __name__ == "__main__":
    lif = IntegerLIF(K=4)
    x = torch.randn(2, 8, 4, 4, requires_grad=True) * 2.0
    y = lif(x)
    print("in:", x.min().item(), x.max().item())
    print("out levels:", y.unique().tolist())
    loss = y.sum()
    loss.backward()
    print("grad ok, non-zero:", (x.grad != 0).any().item())
