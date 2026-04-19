import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt

from data import DataModuleCIFAR10

# --- Model Definition ---

class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta=0.95):
        super(SNN, self).__init__()
        self.num_steps = num_steps

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)


















# --- Demo (single batch visualisation) ---

if __name__ == "__main__":
    data_module = DataModuleCIFAR10(batch_size=128, num_steps=100)
    train_loader, _ = data_module.get_dataloaders(subset=10)

    data_it, targets_it = next(iter(train_loader))

    # Rate-coded spike data
    spike_data_rate = data_module.encode_rate(data_it, gain=1.0)
    spike_data_rate2 = data_module.encode_rate(data_it, gain=0.25)
    spike_data_sample = spike_data_rate[:, 0, 0]
    spike_data_sample2 = spike_data_rate2[:, 0, 0]

    # Latency-coded spike data
    spike_data_latency = data_module.encode_latency(data_it)

    print(f"Rate spike data size: {spike_data_rate.size()}")
    print(f"Latency spike data size: {spike_data_latency.size()}")

    # Latency curve
    raw_input = torch.arange(0.05, 5, 0.05)
    spike_times = data_module.convert_to_time(raw_input)

    plt.plot(raw_input, spike_times)
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.show()

    # Rate coding comparison
    plt.figure(facecolor="w")
    plt.subplot(1, 2, 1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((32, -1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 1')

    plt.subplot(1, 2, 2)
    plt.imshow(spike_data_sample2.mean(axis=0).reshape((32, -1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 0.25')

    plt.show()
