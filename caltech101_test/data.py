from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset


EXCLUDE_CLASSES = {"BACKGROUND_Google"}
MAX_CLASSES = 5


def get_selected_classes(data_path, max_classes=MAX_CLASSES):
    classes = sorted(
        [
            c
            for c in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, c)) and c not in EXCLUDE_CLASSES
        ],
        key=str.casefold,
    )
    return classes[:max_classes]


def load_event_file(filepath):
    """Load N-Caltech101 .bin event file."""
    with open(filepath, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    num_events = len(raw) // 5
    raw = raw[: num_events * 5].reshape(-1, 5)

    x = raw[:, 0].astype(np.int32)
    y = raw[:, 1].astype(np.int32)

    combined = (
        (raw[:, 2].astype(np.int32) << 16)
        | (raw[:, 3].astype(np.int32) << 8)
        | raw[:, 4].astype(np.int32)
    )
    p = ((combined >> 23) & 1).astype(np.int32)
    t = (combined & 0x7FFFFF).astype(np.int32)

    return x, y, p, t


def events_to_frames(x, y, p, t, num_steps=100, height=180, width=240):
    """Bin events into frames of shape (num_steps, 2, H, W)."""
    frames = np.zeros((num_steps, 2, height, width), dtype=np.float32)

    if len(t) == 0:
        return frames

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        return frames

    bins = ((t - t_min) / (t_max - t_min) * (num_steps - 1)).astype(int)
    bins = np.clip(bins, 0, num_steps - 1)

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    p = np.clip(p, 0, 1)

    np.add.at(frames, (bins, p, y, x), 1.0)
    return frames


class NCaltech101TestDataset(Dataset):
    def __init__(self, data_path, num_steps=100, selected_classes=None, balance_classes=True):
        self.num_steps = num_steps
        self.samples = []
        self.labels = []

        available_classes = get_selected_classes(data_path)
        self.selected_classes = selected_classes or available_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.selected_classes)}

        class_files = {cls: [] for cls in self.selected_classes}
        for cls in self.selected_classes:
            cls_path = os.path.join(data_path, cls)
            for fname in sorted(os.listdir(cls_path)):
                if fname.endswith(".bin"):
                    class_files[cls].append(os.path.join(cls_path, fname))

        if balance_classes:
            min_samples = min(len(files) for files in class_files.values())
            print(f"Balancing dataset: capping all classes to {min_samples} samples.")
            for cls in self.selected_classes:
                class_files[cls] = class_files[cls][:min_samples]

        for cls in self.selected_classes:
            for f in class_files[cls]:
                self.samples.append(f)
                self.labels.append(self.class_to_idx[cls])

        print(
            f"Found {len(self.samples)} samples across "
            f"{len(self.selected_classes)} classes: {self.selected_classes}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, p, t = load_event_file(self.samples[idx])
        frames = events_to_frames(x, y, p, t, self.num_steps)
        return torch.tensor(frames), self.labels[idx]


class DataModuleNCaltech101Test:
    def __init__(self, batch_size=32, data_path=None, num_steps=100, seed=42, balance_classes=True):
        project_root = Path(__file__).resolve().parent.parent
        default_data_path = project_root / "data" / "ncaltech101" / "caltech101"

        self.batch_size = batch_size
        self.data_path = str(data_path or default_data_path)
        self.num_steps = num_steps
        self.seed = seed
        self.balance_classes = balance_classes
        self.selected_classes = get_selected_classes(self.data_path)

    def get_dataloaders(self, train_split=0.8, subset=None):
        dataset = NCaltech101TestDataset(
            self.data_path,
            self.num_steps,
            selected_classes=self.selected_classes,
            balance_classes=self.balance_classes,
        )

        if subset is not None:
            class_indices = defaultdict(list)
            for idx, label in enumerate(dataset.labels):
                class_indices[label].append(idx)
            n_per_class = max(1, len(dataset) // (subset * len(class_indices)))
            selected = []
            for indices in class_indices.values():
                selected.extend(indices[:n_per_class])
            dataset = Subset(dataset, selected)

        n_train = int(len(dataset) * train_split)
        n_test = len(dataset) - n_train
        generator = torch.Generator().manual_seed(self.seed)
        train_set, test_set = random_split(
            dataset, [n_train, n_test], generator=generator
        )

        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        return train_loader, test_loader
