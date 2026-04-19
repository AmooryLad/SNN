import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data import DataModuleCIFAR10
from model import SNN

# --- Setup ---

num_steps = 100
num_inputs = 3 * 32 * 32   # flattened CIFAR-10 RGB image
num_hidden = 1000
num_outputs = 10       # CIFAR-10 classes
beta = 0.95
num_epochs = 10
dtype = torch.float

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---

data_module = DataModuleCIFAR10(batch_size=128, num_steps=num_steps)
train_loader, test_loader = data_module.get_dataloaders(subset=10)
batch_size = data_module.batch_size

# --- Model ---

net = SNN(num_inputs, num_hidden, num_outputs, num_steps, beta).to(device)

# 7.2 Loss Definition
loss = nn.CrossEntropyLoss()

# 7.3 Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# Resume from checkpoint if available
start_epoch = 0
best_acc = 0.0
checkpoint_path = 'best_model.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['accuracy']
    print(f"Resumed from epoch {checkpoint['epoch']} (accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found, starting fresh.")

# 7.5 Training Loop
loss_hist = []
test_loss_hist = []
counter = 0
best_epoch = start_epoch

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[-1]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[-1]:.2f}")
    print()

# Outer training loop
for epoch in range(start_epoch, start_epoch + num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

    # Accuracy at end of each epoch
    epoch_correct = 0
    epoch_total = 0
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

    # Save best model by accuracy
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
        }, checkpoint_path)
        print(f">> Best model saved at epoch {epoch} (accuracy: {best_acc:.2f}%)")

print(f"\nBest epoch: {best_epoch} | Best accuracy: {best_acc:.2f}%")

# Load best model for evaluation
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
print("Loaded best model weights for evaluation.")

# 8.2 Test Set Accuracy
total = 0
correct = 0

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        test_spk, _ = net(data.view(data.size(0), -1))
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
