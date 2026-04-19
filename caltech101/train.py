import sys
from pathlib import Path

from snntorch import surrogate
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from caltech101.model import SNNConvClassifier
from caltech101.data import DataModuleNCaltech101

spike_grad = surrogate.fast_sigmoid()
EXPERIMENT_NAME = "caltech101_conv_snn"
SEED = 42
num_steps = 50           # halved from 100 — fewer steps = less inference energy
batch_size = 16
beta = 0.95
num_epochs = 40
lr = 1e-3
weight_decay = 1e-4
spike_reg_weight = 1e-3  # L1 penalty on mean firing rate (energy proxy)
label_smoothing = 0.1

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = DataModuleNCaltech101(
    batch_size=batch_size,
    data_path=str(PROJECT_ROOT / "data" / "ncaltech101" / "caltech101"),
    num_steps=num_steps,
    seed=SEED,
)
train_loader, test_loader = data_module.get_dataloaders(subset=None)
num_classes = len(data_module.class_names)

# Class-weighted loss: inverse frequency normalised so mean weight = 1
train_subset = train_loader.dataset
train_labels = torch.tensor(
    [train_subset.dataset.labels[i] for i in train_subset.indices]
)
class_counts = torch.bincount(train_labels, minlength=num_classes).float()
class_weights = class_counts.sum() / (num_classes * class_counts.clamp(min=1))
class_weights = class_weights.to(device)

print(f"Training on {num_classes} classes")
print(f"Experiment: {EXPERIMENT_NAME} | Seed: {SEED} | Batch size: {batch_size}")
print(f"Class count range: {int(class_counts.min())}-{int(class_counts.max())}")

net = SNNConvClassifier(
    num_classes=num_classes,
    num_steps=num_steps,
    beta=beta,
    spike_grad=spike_grad,
).to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
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

# --- Training ---

for epoch in range(start_epoch, start_epoch + num_epochs):
    net.train()
    running_ce, running_spike, n_batches = 0.0, 0.0, 0
    for events, targets in train_loader:
        # events: [B, T, 2, H, W] -> [T, B, 2, H, W]
        events = events.float().to(device).transpose(0, 1)
        targets = targets.to(device)

        spk_out_rec, fire_rates = net(events)
        logits = spk_out_rec.sum(dim=0)            # rate coding
        ce_loss = loss_fn(logits, targets)
        spike_loss = sum(fire_rates)               # encourage sparsity
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
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    eval_fire_rates = [0.0, 0.0, 0.0, 0.0]
    eval_batches = 0
    net.eval()
    with torch.no_grad():
        for events, targets in test_loader:
            events = events.float().to(device).transpose(0, 1)
            targets = targets.to(device)
            spk_out_rec, fire_rates = net(events)
            logits = spk_out_rec.sum(dim=0)
            _, predicted = logits.max(1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            for target, pred in zip(targets, predicted):
                class_total[target.item()] += 1
                if pred.item() == target.item():
                    class_correct[target.item()] += 1
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
        "  Fire rates (lif1/lif2/lif3/out): "
        + " ".join(f"{r:.3f}" for r in mean_fire)
    )
    for class_idx, class_name in enumerate(data_module.class_names):
        total = class_total[class_idx]
        if total == 0:
            continue
        class_acc = 100 * class_correct[class_idx] / total
        print(
            f"  {class_name:<18} {class_correct[class_idx]}/{total} "
            f"({class_acc:.2f}%)"
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
        }, checkpoint_path)
        print(f">> Best saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")
