"""
Transfer learning: load a backbone trained on one dataset, fine-tune on another.

Example: train backbone on N-Caltech101 (neuromorphic), then fine-tune on CIFAR-10.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.snn import SNNClassifier
from cifar10.data import DataModuleCIFAR10

# --- Config ---

SOURCE_BACKBONE = '../checkpoints/caltech101_backbone.pth'  # where we load from
TARGET_CHECKPOINT = '../checkpoints/cifar10_from_caltech.pth'  # where we save to
FREEZE_BACKBONE = False  # set True to only train the head

num_steps = 100
num_inputs = 3 * 32 * 32
num_hidden = 1000
num_classes = 10
beta = 0.95
num_epochs = 10
dtype = torch.float

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---

data_module = DataModuleCIFAR10(batch_size=128, num_steps=num_steps)
train_loader, test_loader = data_module.get_dataloaders(subset=10)
batch_size = data_module.batch_size

# --- Model with transferred backbone ---

net = SNNClassifier(num_inputs, num_hidden, num_classes, num_steps, beta).to(device)

if os.path.exists(SOURCE_BACKBONE):
    # NOTE: backbone weights only transfer if num_inputs matches.
    # For different input sizes (e.g. Caltech's 86400 → CIFAR's 3072), fc1 must be
    # re-initialised. Here we assume matching shapes or skip fc1 manually.
    backbone_state = torch.load(SOURCE_BACKBONE)
    try:
        net.backbone.load_state_dict(backbone_state)
        print(f"Loaded backbone from {SOURCE_BACKBONE}")
    except RuntimeError as e:
        # Input sizes differ — load everything except fc1
        filtered = {k: v for k, v in backbone_state.items() if not k.startswith('fc1')}
        net.backbone.load_state_dict(filtered, strict=False)
        print(f"Loaded partial backbone (skipped fc1 — input size mismatch)")
else:
    print(f"Warning: no source backbone at {SOURCE_BACKBONE} — training from scratch")

if FREEZE_BACKBONE:
    for param in net.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen — only head will train")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=5e-4, betas=(0.9, 0.999)
)

# --- Fine-tuning loop ---

best_acc = 0.0
for epoch in range(num_epochs):
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

    # Epoch accuracy
    correct, total = 0, 0
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            test_spk, _ = net(data.view(data.size(0), -1))
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = 100 * correct / total
    print(f"Epoch {epoch} | Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
        }, TARGET_CHECKPOINT)
        print(f">> Best saved (accuracy: {best_acc:.2f}%)")

print(f"\nFinal best accuracy: {best_acc:.2f}%")
