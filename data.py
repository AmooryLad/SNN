import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils, spikegen


class DataModuleCIFAR10:
    def __init__(self, batch_size=128, data_path='./data/cifar-10', num_classes=10, num_steps=100):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_classes = num_classes
        self.num_steps = num_steps

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def get_dataloaders(self, subset=10):
        cifar_train = datasets.CIFAR10(self.data_path, train=True, download=True, transform=self.transform)
        cifar_test = datasets.CIFAR10(self.data_path, train=False, download=True, transform=self.transform)

        cifar_train.targets = np.array(cifar_train.targets)
        cifar_test.targets = np.array(cifar_test.targets)

        cifar_train = utils.data_subset(cifar_train, subset)
        cifar_test = utils.data_subset(cifar_test, subset)

        print(f"Training set size: {len(cifar_train)}")
        print(f"Test set size: {len(cifar_test)}")

        train_loader = DataLoader(cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(cifar_test, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return train_loader, test_loader

    def encode_rate(self, data, gain=1.0):
        return spikegen.rate(data, num_steps=self.num_steps, gain=gain)

    def encode_latency(self, data, tau=5, threshold=0.01):
        return spikegen.latency(data, num_steps=self.num_steps, tau=tau, threshold=threshold)

    def convert_to_time(self, data, tau=5, threshold=0.01):
        return tau * torch.log(data / (data - threshold))
