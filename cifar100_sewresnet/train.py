import copy
import sys
from pathlib import Path

import numpy as np
from snntorch import surrogate
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from cifar100_sewresnet.model import SEWResNet18
from cifar100_sewresnet.data import DataModuleCIFAR100


# --- Experiment selector -------------------------------------------------
# SEW-ResNet-18 on CIFAR-100.
#   S: base recipe, 80 ep cosine (final: 72.99%)
#   T: longer schedule + warmup + Mixup/CutMix + EMA (target: 75-77%)

EXPERIMENT = 'T'

EXPERIMENTS = {
    'S':  (8, 'strong',  80,  0,  False, 'SEW + TET + ATan + strong aug (baseline)'),
    'T':  (8, 'strong', 120, 10, True,  'S + 120ep + warmup10 + Mixup/CutMix + EMA'),
}

num_steps, aug_level, num_epochs, warmup_epochs, use_mix_ema, exp_desc = \
    EXPERIMENTS[EXPERIMENT]
EXPERIMENT_NAME = f"cifar100_sew_{EXPERIMENT}"

# --- Hyperparameters -----------------------------------------------------

SEED = 42
batch_size = 128
beta = 0.95
lr = 1e-3
weight_decay = 5e-4
label_smoothing = 0.1
spike_reg_weight = 1e-3
spike_grad = surrogate.atan()
mix_prob = 0.5          # probability of applying mixup-or-cutmix per batch
mixup_alpha = 0.2
cutmix_alpha = 1.0
ema_decay = 0.999

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Log tee -------------------------------------------------------------

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"{EXPERIMENT_NAME}.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(log_path, 'w', buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"=== Experiment {EXPERIMENT}: {exp_desc} ===")
print(f"Log: {log_path}")
print(f"num_steps={num_steps} aug={aug_level} epochs={num_epochs} "
      f"warmup={warmup_epochs} mix/ema={use_mix_ema}")

# --- Data ----------------------------------------------------------------

data_module = DataModuleCIFAR100(
    batch_size=batch_size,
    data_path=str(PROJECT_ROOT / "data" / "cifar-100"),
    aug_level=aug_level,
)
train_loader, test_loader = data_module.get_dataloaders()
num_classes = len(data_module.class_names)

print(f"Training on {num_classes} classes")
print(f"Experiment: {EXPERIMENT_NAME} | Seed: {SEED}")

# --- Mixup / CutMix ------------------------------------------------------


def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def cutmix(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bx1, by1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    bx2, by2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    x_new = x.clone()
    x_new[:, :, by1:by2, bx1:bx2] = x[idx, :, by1:by2, bx1:bx2]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (H * W))
    return x_new, y, y[idx], lam


# --- EMA ----------------------------------------------------------------


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.ema.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        # Sync BN running stats. snn.Leaky lazily allocates mem/state buffers
        # on first forward, so shapes won't line up until then — reassign in
        # that case instead of in-place copy.
        for e, b in zip(self.ema.buffers(), model.buffers()):
            if e.shape == b.shape and e.dtype == b.dtype:
                e.copy_(b)
            else:
                e.data = b.data.clone()


# --- Model ---------------------------------------------------------------

net = SEWResNet18(
    num_classes=num_classes,
    num_steps=num_steps,
    beta=beta,
    spike_grad=spike_grad,
).to(device)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")

ema = ModelEMA(net, decay=ema_decay) if use_mix_ema else None

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

# Linear warmup -> cosine annealing
if warmup_epochs > 0:
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs],
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / f"{EXPERIMENT_NAME}_best.pth"
start_epoch = 0
best_acc = 0.0

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.ema.load_state_dict(checkpoint['ema_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['accuracy']
    print(f"Resumed from epoch {checkpoint['epoch']} (accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found, starting fresh.")

fire_labels = ["stem", "layer1", "layer2", "layer3", "layer4", "out"]

# --- Training ------------------------------------------------------------

for epoch in range(start_epoch, start_epoch + num_epochs):
    net.train()
    running_ce, running_spike, n_batches = 0.0, 0.0, 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        # Mixup or CutMix (50/50) with probability mix_prob
        if use_mix_ema and np.random.rand() < mix_prob:
            if np.random.rand() < 0.5:
                images, y_a, y_b, lam = mixup(images, targets, mixup_alpha)
            else:
                images, y_a, y_b, lam = cutmix(images, targets, cutmix_alpha)
        else:
            y_a, y_b, lam = targets, targets, 1.0

        spk_out_rec, mem_out, fire_rates, logit_rec = net(images)
        T = logit_rec.size(0)
        # TET loss with mixed labels
        ce_loss = sum(
            lam * loss_fn(logit_rec[t], y_a) + (1 - lam) * loss_fn(logit_rec[t], y_b)
            for t in range(T)
        ) / T
        spike_loss = sum(fire_rates)
        loss_val = ce_loss + spike_reg_weight * spike_loss

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(net)

        running_ce += ce_loss.item()
        running_spike += spike_loss.item()
        n_batches += 1

    scheduler.step()

    # --- Eval (use EMA weights if available) -----------------------------
    eval_net = ema.ema if ema is not None else net
    eval_net.eval()
    epoch_correct, epoch_total = 0, 0
    eval_fire_rates = [0.0] * 6
    eval_batches = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            spk_out_rec, mem_out, fire_rates, logit_rec = eval_net(images)
            logits = logit_rec.mean(dim=0)
            _, predicted = logits.max(1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            for i, fr in enumerate(fire_rates):
                eval_fire_rates[i] += fr.item()
            eval_batches += 1

    epoch_acc = 100 * epoch_correct / epoch_total
    mean_fire = [fr / eval_batches for fr in eval_fire_rates]

    tag = "EMA" if ema is not None else "raw"
    print(
        f"Epoch {epoch} | Acc({tag}): {epoch_acc:.2f}% | "
        f"CE: {running_ce / n_batches:.3f} | SpikeReg: {running_spike / n_batches:.3f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )
    print(
        f"  Fire rates ({'/'.join(fire_labels)}): "
        + " ".join(f"{r:.3f}" for r in mean_fire)
    )

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state_dict': ema.ema.state_dict() if ema is not None else None,
            'accuracy': best_acc,
            'classes': data_module.class_names,
            'experiment_name': EXPERIMENT_NAME,
            'seed': SEED,
            'num_steps': num_steps,
        }, checkpoint_path)
        print(f">> Best saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")
