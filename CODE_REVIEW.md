# Code Review: Caltech101 SNN Training Pipeline

**Reviewer:** Senior ML Engineer (SNN specialist), in a bad mood
**Subject:** Architecture and training pipeline written by GPT-5.4
**Verdict:** *Sigh.*

---

## TL;DR

This model isn't learning — it's memorizing class priors. Look at the epoch 2 output:

```
airplanes       158/160 (98.75%)    ← 160 samples = easy
Motorbikes      157/160 (98.12%)    ← 160 samples = easy
Faces_easy       83/87  (95.40%)    ← 87 samples = easy
car_side         25/25  (100.00%)   ← distinctive background
Leopards         36/40  (90.00%)    ← distinctive texture
...
most other classes: 0%
```

The model has learned "if it has many samples or a unique background, predict that. Otherwise, don't bother." That's not learning. That's a histogram with delusions of grandeur.

---

## Critical Issues

### 1. You're using a FULLY CONNECTED network on EVENT-BASED VISION data

**The single biggest problem.** You are flattening `(100, 2, 180, 240) → 86,400` inputs and feeding that into an FC layer. You threw away every bit of spatial structure the event camera captured. A leopard's rosettes? Spatial. A motorcycle's frame? Spatial. A crab's legs? Spatial. Your model sees none of it.

**What you should have:** convolutional SNN layers. `snn.Conv2d` + `snn.Leaky`, preferably with pooling. The fact that this architecture gets 34% at all is a miracle, not a success — it's learning dominant color/density patterns across the whole visual field, not objects.

### 2. Loss summed over ALL time steps

```python
for step in range(num_steps):
    loss_val += loss(mem_rec[step], targets)
```

You're asking the model to classify correctly **at every single one of 100 time steps**, including step 0 when it has seen literally no input. This is nonsense. Early time steps have near-zero membrane potential — CE loss there is basically "be uniform about random noise," which just pushes the final layer biases around.

**Fix:** Either
- Compute loss only on `mem_rec[-1]` (final prediction)
- Use spike count: `loss(spk_rec.sum(dim=0), targets)` and train with rate coding
- Weight the temporal loss (loss later = more important)

### 3. No surrogate gradient was set initially (now partially fixed)

You added `spike_grad=surrogate.fast_sigmoid()` late in the game. Before that, the model was using snntorch's default `atan` surrogate, which is fine but the *inconsistency* across checkpoints means your 512-hidden and 1000-hidden experiments aren't comparable.

### 4. No regularization

Zero dropout, zero weight decay, zero data augmentation. You have 86M parameters in `fc1` alone (`86400 × 1000`) and 6,559 training samples. That's **over 13,000 parameters per training sample.** It WILL overfit. It's a matter of when, not if.

**Fix:**
- `dropout` between `fc1→fc2` and before head
- `weight_decay=1e-4` in optimizer
- Event-specific augmentation: spatial jitter, polarity flip, time reversal

### 5. Severe class imbalance, zero mitigation

```
airplanes: 160 test samples
garfield:    7 test samples
```

That's a 22× imbalance. The model just learned class priors. Your CrossEntropyLoss sees 20× more gradient signal from airplanes than garfield per epoch. Of course garfield accuracy is 0%.

**Fix:**
- `CrossEntropyLoss(weight=class_weights)` where weights are inverse frequency
- Or use `WeightedRandomSampler` in the DataLoader
- Or oversample minority classes

### 6. Batch size = 8 is way too small

With 8 samples per batch across 100 classes, most batches don't contain most classes. Gradient updates become extremely noisy. This is why your per-class accuracy fluctuates wildly between epochs (anchor was 0% at epoch 0, still 0% at epoch 2 — never gets any gradient because it's rarely in a batch).

**Fix:** Increase to `batch_size=32` or `64`. If VRAM is the issue, switch to convolutions (which use less memory *and* are more effective).

### 7. No learning rate scheduler

Adam at a fixed `3e-4` for 30 epochs. You'll either overshoot minima or crawl near the end. Add a `CosineAnnealingLR` or `StepLR`.

### 8. Membrane potentials reset between samples but not between time steps within a forward pass

Each forward pass starts with `mem=0`. For static images, fine. For N-Caltech101 (event data with real temporal structure from saccades), this loses information about which saccade direction you're in. But then again you threw away spatial structure too, so what's one more sin.

### 9. `beta=0.95` hardcoded and shared

Decay rate is a hyperparameter that SHOULD vary per layer (deeper layers often benefit from slower decay). At minimum, make it learnable:

```python
snn.Leaky(beta=beta, learn_beta=True)
```

### 10. Threshold never tuned

Default `threshold=1.0` for all layers. For data with highly variable event densities, this threshold might be too high (neurons never fire) or too low (everything fires, no selectivity). Either tune it or make it learnable: `learn_threshold=True`.

### 11. No input normalization for events

You bin events into frames and feed raw counts. Some time bins have thousands of events, others have zero. Input magnitude varies wildly between samples. At minimum, normalize per-sample (divide by max count or use log scaling).

### 12. Fixed 100 time steps regardless of event duration

Some N-Caltech101 samples span hundreds of ms, others span tens. Binning both into 100 uniform steps means your temporal resolution varies sample-to-sample and your network can't learn invariance to that.

---

## What I Would Rewrite

```python
class SNNConvClassifier(nn.Module):
    def __init__(self, num_classes, num_steps, beta=0.95, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps

        # Spatial features preserved
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                 learn_beta=True, learn_threshold=True)

    def forward(self, x):  # x: [T, B, 2, H, W]
        # ... proper temporal forward with all state resets
        # Accumulate spikes at output, train on final rate
        return spk_out_rec.sum(dim=0)  # rate-based output
```

And the training loop:

```python
# Class weights
class_counts = torch.tensor([count_per_class[i] for i in range(num_classes)])
class_weights = (class_counts.sum() / (num_classes * class_counts)).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Loss on accumulated spike count, not summed over time
loss_val = loss_fn(spk_out.sum(dim=0), targets)
```

---

## Expected Outcome If Fixed

With FC architecture on N-Caltech101 you will asymptote around 40-50% and then overfit. **This is a hard ceiling no amount of learning rate tuning will break.**

With a CNN backbone (even a small one), published SNN papers on N-Caltech101 reach 75-85%. That's your target. Current setup will never get there.

---

## Summary of Priority Fixes

| # | Fix | Effort | Expected gain |
|---|---|---|---|
| 1 | Replace FC with Conv SNN | High | +30-40% |
| 2 | Loss on final step / spike count, not sum | Low | +5-10% |
| 3 | Class-weighted loss | Low | +5% (rare classes) |
| 4 | Dropout + weight decay | Low | +3-5% (less overfit) |
| 5 | Learn beta + threshold | Low | +2-5% |
| 6 | LR scheduler | Low | +2-3% |
| 7 | Larger batch size | Low | Training stability |
| 8 | Event augmentation | Medium | +3-5% |

Do #1 first. Everything else is a rounding error compared to that.

---

## Closing Remarks

The code is *clean*. That's genuinely nice. The folder structure (backbone / heads / dataset modules) is sensible. Checkpoint resume logic works. The author clearly had good software engineering instincts.

But they treated this like a generic classification problem when it's a neuromorphic vision problem, and they used the same architecture you'd use for MNIST. MNIST is 784 pixels. You have 86,400 inputs AND temporal structure AND spatial structure, and this architecture throws away the last two.

Fix the architecture. Then come back.
