import sys
from pathlib import Path

from snntorch import surrogate
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from caltech101_test.data import DataModuleNCaltech101Test
from models.snn import SNNClassifier

spike_grad = surrogate.fast_sigmoid()

EXPERIMENT_NAME = "caltech101_test_fast_sigmoid"
SEED = 42
num_steps = 100
num_inputs = 2 * 180 * 240
num_hidden = 1000
beta = 0.95
num_epochs = 10
dtype = torch.float

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = DataModuleNCaltech101Test(batch_size=32, num_steps=num_steps, seed=SEED)
train_loader, test_loader = data_module.get_dataloaders()
num_classes = len(data_module.selected_classes)
batch_size = data_module.batch_size

print(f"Training only on these classes: {data_module.selected_classes}")
print(f"Experiment: {EXPERIMENT_NAME} | Seed: {SEED}")

net = SNNClassifier(
    num_inputs,
    num_hidden,
    num_classes,
    num_steps,
    beta,
    spike_grad=spike_grad,
).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / f"{EXPERIMENT_NAME}_best.pth"
backbone_path = checkpoint_dir / f"{EXPERIMENT_NAME}_backbone.pth"

start_epoch = 0
best_acc = 0.0

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["accuracy"]
    print(f"Resumed from epoch {checkpoint['epoch']} (accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found, starting fresh.")

for epoch in range(start_epoch, start_epoch + num_epochs):
    for events, targets in train_loader:
        events = events.float().to(device)
        targets = targets.to(device)

        events = events.transpose(0, 1)
        events_flat = events.reshape(num_steps, events.size(1), -1)

        net.train()
        spk_rec, mem_rec = net(None, x_seq=events_flat)

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

    epoch_correct, epoch_total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        net.eval()
        for events, targets in test_loader:
            events = events.float().to(device)
            targets = targets.to(device)
            events = events.transpose(0, 1)
            events_flat = events.reshape(num_steps, events.size(1), -1)
            test_spk, _ = net(None, x_seq=events_flat)
            _, predicted = test_spk.sum(dim=0).max(1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            for target, pred in zip(targets, predicted):
                class_total[target.item()] += 1
                if pred.item() == target.item():
                    class_correct[target.item()] += 1
    epoch_acc = 100 * epoch_correct / epoch_total
    print(f"Epoch {epoch} | Accuracy: {epoch_acc:.2f}%")
    for class_idx, class_name in enumerate(data_module.selected_classes):
        class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
        print(
            f"  {class_name:<12} {class_correct[class_idx]}/{class_total[class_idx]} "
            f"({class_acc:.2f}%)"
        )

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
                "classes": data_module.selected_classes,
                "experiment_name": EXPERIMENT_NAME,
                "seed": SEED,
            },
            checkpoint_path,
        )
        torch.save(net.backbone.state_dict(), backbone_path)
        print(f">> Best saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")

print(f"\nFinal best accuracy: {best_acc:.2f}%")
