import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add project root so we can import `models`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.snn import SNNClassifier
from cifar10.data import DataModuleCIFAR10

# --- Setup ---

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

# --- Model ---

net = SNNClassifier(num_inputs, num_hidden, num_classes, num_steps, beta).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# --- Resume from checkpoint ---

checkpoint_path = '../checkpoints/cifar10_best.pth'
start_epoch = 0
best_acc = 0.0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['accuracy']
    print(f"Resumed from epoch {checkpoint['epoch']} (accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found, starting fresh.")

# --- Training ---

loss_hist = []
test_loss_hist = []
counter = 0
best_epoch = start_epoch

for epoch in range(start_epoch, start_epoch + num_epochs):
    iter_counter = 0

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

        loss_hist.append(loss_val.item())

        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            test_spk, test_mem = net(test_data.view(batch_size, -1))

            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            counter += 1
            iter_counter += 1

    # Epoch accuracy
    epoch_correct, epoch_total = 0, 0
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            test_spk, _ = net(data.view(data.size(0), -1))
            _, predicted = test_spk.sum(dim=0).max(1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
    epoch_acc = 100 * epoch_correct / epoch_total
    print(f"Epoch {epoch} | Accuracy: {epoch_acc:.2f}%")

    # Save best (full model + backbone separately)
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
        }, checkpoint_path)
        # Save backbone alone for transfer learning
        torch.save(net.backbone.state_dict(), '../checkpoints/cifar10_backbone.pth')
        print(f">> Best saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")

print(f"\nBest epoch: {best_epoch} | Best accuracy: {best_acc:.2f}%")

# Plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("CIFAR-10 Loss Curves")
plt.legend(["Train", "Test"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
