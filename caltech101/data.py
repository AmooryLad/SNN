from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


def load_event_file(filepath):
    """Load N-Caltech101 .bin event file.

    Format: 40 bits per event (5 bytes)
    - bits 39-32: X address (8 bits)
    - bits 31-24: Y address (8 bits)
    - bit 23: Polarity (1 bit)
    - bits 22-0: Timestamp (23 bits)
    """
    with open(filepath, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    # Each event is 5 bytes
    num_events = len(raw) // 5
    raw = raw[:num_events * 5].reshape(-1, 5)

    # Parse 5 bytes into 40-bit values
    # Byte order (big-endian): X(8) Y(8) P+T(24)
    x = raw[:, 0].astype(np.int32)
    y = raw[:, 1].astype(np.int32)

    # Polarity + timestamp in last 3 bytes
    combined = ((raw[:, 2].astype(np.int32) << 16) |
                (raw[:, 3].astype(np.int32) << 8) |
                raw[:, 4].astype(np.int32))
    p = ((combined >> 23) & 1).astype(np.int32)
    t = (combined & 0x7FFFFF).astype(np.int32)

    return x, y, p, t


def events_to_frames(x, y, p, t, num_steps=100, height=180, width=240):
    """Bin events into num_steps frames of shape (num_steps, 2, H, W)."""
    frames = np.zeros((num_steps, 2, height, width), dtype=np.float32)

    if len(t) == 0:
        return frames

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        return frames

    # Assign each event to a time bin
    bins = ((t - t_min) / (t_max - t_min) * (num_steps - 1)).astype(int)
    bins = np.clip(bins, 0, num_steps - 1)

    # Clip coordinates to valid range
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    p = np.clip(p, 0, 1)

    np.add.at(frames, (bins, p, y, x), 1.0)
    return frames


class NCaltech101Dataset(Dataset):
    def __init__(self, data_path, num_steps=100):
        self.num_steps = num_steps
        self.samples = []
        self.labels = []

        # Exclude BACKGROUND_Google — not a real object class
        EXCLUDE = {'BACKGROUND_Google'}
        self.classes = sorted(
            [
                c for c in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, c)) and c not in EXCLUDE
            ],
            key=str.casefold,
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(data_path, cls)
            for fname in sorted(os.listdir(cls_path)):
                if fname.endswith('.bin'):
                    self.samples.append(os.path.join(cls_path, fname))
                    self.labels.append(self.class_to_idx[cls])

        print(f"Found {len(self.samples)} samples across {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, p, t = load_event_file(self.samples[idx])
        frames = events_to_frames(x, y, p, t, self.num_steps)
        return torch.tensor(frames), self.labels[idx]


class AugmentedEventDataset(Dataset):
    """Wraps an event dataset to apply train-time augmentations to binned frames.

    Augmentations operate on the (T, 2, H, W) frame tensor:
    - Random horizontal flip (p=0.5) — reflects events along width.
    - Polarity swap (p=0.5) — swaps ON/OFF channels, equivalent to contrast flip.
    - Random crop with zero-padding — pad by `crop_pad` on all sides, then
      crop back to (H, W) at a random offset. Preserves event locality while
      jittering spatial position.
    """

    def __init__(self, base, augment=True, crop_pad=16,
                 flip_prob=0.5, polarity_prob=0.5):
        self.base = base
        self.augment = augment
        self.crop_pad = crop_pad
        self.flip_prob = flip_prob
        self.polarity_prob = polarity_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, label = self.base[idx]
        if self.augment:
            frames = self._augment(frames)
        return frames, label

    def _augment(self, frames):
        if torch.rand(1).item() < self.flip_prob:
            frames = torch.flip(frames, dims=[-1])

        if torch.rand(1).item() < self.polarity_prob:
            frames = frames[:, [1, 0], :, :]

        pad = self.crop_pad
        if pad > 0:
            T, C, H, W = frames.shape
            padded = F.pad(frames, (pad, pad, pad, pad))
            top = torch.randint(0, 2 * pad + 1, (1,)).item()
            left = torch.randint(0, 2 * pad + 1, (1,)).item()
            frames = padded[:, :, top:top + H, left:left + W]

        return frames


class DataModuleNCaltech101:
    def __init__(self, batch_size=32, data_path='./data/ncaltech101',
                 num_steps=100, seed=42,
                 augment_train=False, crop_pad=16,
                 flip_prob=0.5, polarity_prob=0.5):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_steps = num_steps
        self.seed = seed
        self.augment_train = augment_train
        self.crop_pad = crop_pad
        self.flip_prob = flip_prob
        self.polarity_prob = polarity_prob
        self.class_names = []

    def get_dataloaders(self, train_split=0.8, subset=None):
        dataset = NCaltech101Dataset(self.data_path, self.num_steps)
        self.class_names = dataset.classes
        selected_indices = list(range(len(dataset)))

        if subset is not None:
            # Stratified subset: take equal samples per class
            class_indices = defaultdict(list)
            for idx, label in enumerate(dataset.labels):
                class_indices[label].append(idx)
            n_per_class = max(1, len(dataset) // (subset * len(class_indices)))
            selected_indices = []
            for indices in class_indices.values():
                selected_indices.extend(indices[:n_per_class])

        generator = torch.Generator().manual_seed(self.seed)
        split_indices = defaultdict(list)
        for idx in selected_indices:
            split_indices[dataset.labels[idx]].append(idx)

        train_indices = []
        test_indices = []
        for indices in split_indices.values():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            n_train = int(len(shuffled) * train_split)
            n_train = max(1, min(n_train, len(shuffled) - 1))
            train_indices.extend(shuffled[:n_train])
            test_indices.extend(shuffled[n_train:])

        train_set = Subset(dataset, train_indices)
        test_set = Subset(dataset, test_indices)

        if self.augment_train:
            train_set = AugmentedEventDataset(
                train_set, augment=True, crop_pad=self.crop_pad,
                flip_prob=self.flip_prob, polarity_prob=self.polarity_prob,
            )

        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")
        if self.augment_train:
            print(
                f"Train augmentation: flip(p={self.flip_prob}) + "
                f"polarity(p={self.polarity_prob}) + crop(pad={self.crop_pad})"
            )

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True, num_workers=16, pin_memory=True,
                          persistent_workers=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size,
                                 shuffle=False, drop_last=False, num_workers=8, pin_memory=True,
                         persistent_workers=True)

        return train_loader, test_loader
