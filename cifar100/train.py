import sys
from pathlib import Path

from snntorch import surrogate
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from cifar100.model import SNNConvClassifier
from cifar100.data import DataModuleCIFAR100


# --- Experiment selector -------------------------------------------------
# CIFAR-100 Phase 3: caltech-style interleaved loop, no attention.
# Add improvements one at a time.
#
# Plan:
#   P (pure baseline): caltech-style, no attn, T=8, 60 epochs
#   Q (aug):           P + RandAugment + Cutout
#   R (steps):         Q + T=12
#   S (skip):          R + SEW-ResNet skip connections (model change needed)

EXPERIMENT = 'P'

EXPERIMENTS = {
    # letter: (num_steps, bn_mode, aug_level, num_epochs, label)
    'P': (8,  'bntt', 'basic',  60,  'Phase 3 baseline: caltech-style, no attn, T=8'),
    'Q': (8,  'bntt', 'strong', 60,  'Phase 3 + RandAugment + Cutout'),
    'R': (12, 'bntt', 'strong', 60,  'Phase 3 + strong aug + T=12'),
}

num_steps, bn_mode, aug_level, num_epochs, exp_desc = EXPERIMENTS[EXPERIMENT]
EXPERIMENT_NAME = f"cifar100_p3_{EXPERIMENT}"

# --- Hyperparameters -----------------------------------------------------

SEED = 42
batch_size = 128
beta = 0.95
lr = 1e-3
weight_decay = 5e-4
label_smoothing = 0.1
spike_reg_weight = 1e-3
spike_grad = surrogate.fast_sigmoid()

torch.manual_seed(SEED)
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
print(f"num_steps={num_steps} bn_mode={bn_mode} aug={aug_level} epochs={num_epochs}")

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

# --- Model ---------------------------------------------------------------

net = SNNConvClassifier(
    num_classes=num_classes,
    num_steps=num_steps,
    beta=beta,
    spike_grad=spike_grad,
    bn_mode=bn_mode,
).to(device)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
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
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['accuracy']
    print(f"Resumed from epoch {checkpoint['epoch']} (accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found, starting fresh.")

fire_labels = ["lif1", "lif2", "lif3", "lif4", "out"]

# --- Training ------------------------------------------------------------

for epoch in range(start_epoch, start_epoch + num_epochs):
    net.train()
    running_ce, running_spike, n_batches = 0.0, 0.0, 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        spk_out_rec, logits, fire_rates = net(images)
        ce_loss = loss_fn(logits, targets)
        spike_loss = sum(fire_rates)
        loss_val = ce_loss + spike_reg_weight * spike_loss

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        running_ce += ce_loss.item()
        running_spike += spike_loss.item()
        n_batches += 1

    scheduler.step()

    # --- Eval ---
    epoch_correct, epoch_total = 0, 0
    eval_fire_rates = [0.0] * 5
    eval_batches = 0
    net.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            spk_out_rec, logits, fire_rates = net(images)
            _, predicted = logits.max(1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            for i, fr in enumerate(fire_rates):
                eval_fire_rates[i] += fr.item()
            eval_batches += 1

    epoch_acc = 100 * epoch_correct / epoch_total
    mean_fire = [fr / eval_batches for fr in eval_fire_rates]

    print(
        f"Epoch {epoch} | Acc: {epoch_acc:.2f}% | "
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
            'accuracy': best_acc,
            'classes': data_module.class_names,
            'experiment_name': EXPERIMENT_NAME,
            'seed': SEED,
            'num_steps': num_steps,
            'bn_mode': bn_mode,
        }, checkpoint_path)
        print(f">> Best saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")
