from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class Cutout:
    """Randomly mask a square patch of the image."""
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        import torch
        _, h, w = img.shape
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1, y2 = max(0, y - self.size // 2), min(h, y + self.size // 2)
        x1, x2 = max(0, x - self.size // 2), min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


class DataModuleCIFAR100:
    """CIFAR-100 loader.

    aug_level:
      'none'   — normalize only
      'basic'  — RandomCrop(pad=4) + HFlip (2015-era baseline)
      'strong' — RandAugment + Cutout + HFlip + RandomCrop
    """

    def __init__(self, batch_size=128, data_path='./data/cifar-100',
                 aug_level='basic', num_workers=8):
        self.batch_size = batch_size
        self.data_path = data_path
        self.aug_level = aug_level
        self.num_workers = num_workers
        self.class_names = []

        normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

        if aug_level == 'none':
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        elif aug_level == 'basic':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif aug_level == 'strong':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                Cutout(size=16),
            ])
        else:
            raise ValueError(f"aug_level must be none/basic/strong, got {aug_level}")

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_dataloaders(self):
        train_set = datasets.CIFAR100(
            self.data_path, train=True, download=True, transform=self.train_transform,
        )
        test_set = datasets.CIFAR100(
            self.data_path, train=False, download=True, transform=self.test_transform,
        )

        self.class_names = train_set.classes

        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")
        print(f"Augmentation: {self.aug_level}")

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, drop_last=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
        )

        return train_loader, test_loader
