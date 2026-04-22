"""Pascal VOC 2012 detection data loader for RetinaNet.

VOC returns (PIL.Image, {'annotation': {...}}); we convert to RetinaNet format
({'boxes': [N,4] tensor in xyxy, 'labels': [N] int tensor}).

Transforms use torchvision.transforms.v2 so bounding boxes are auto-updated
(resize, flip) alongside the image via tv_tensors.BoundingBoxes.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2


# VOC 2012 classes. Index 0 is reserved for background (RetinaNet convention).
VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


class VOCDetectionAdapter(torch.utils.data.Dataset):
    """Wrap torchvision VOCDetection, returning (image_tensor, target_dict)."""

    def __init__(self, root, image_set="train", transforms=None, download=True):
        self.voc = VOCDetection(
            root=str(root), year="2012", image_set=image_set, download=download,
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target_raw = self.voc[idx]
        annotations = target_raw["annotation"]
        W = int(annotations["size"]["width"])
        H = int(annotations["size"]["height"])
        objects = annotations.get("object", [])
        if isinstance(objects, dict):          # single-object samples come back as dict
            objects = [objects]

        boxes, labels = [], []
        for obj in objects:
            name = obj["name"]
            if name not in CLASS_TO_IDX:
                continue
            bb = obj["bndbox"]
            x1 = float(bb["xmin"]); y1 = float(bb["ymin"])
            x2 = float(bb["xmax"]); y2 = float(bb["ymax"])
            # Skip degenerate boxes that would break the regression loss
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[name])

        if len(boxes) == 0:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY", canvas_size=(H, W),
            )
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format="XYXY", canvas_size=(H, W),
            )
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Ensure boxes tensor is a plain FloatTensor (not tv_tensors) for RetinaNet.
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor).float()
        return img, target


def build_transforms(img_size=416, train=True):
    tfms = []
    if train:
        tfms.append(v2.RandomHorizontalFlip(p=0.5))
        tfms.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    tfms.append(v2.Resize((img_size, img_size), antialias=True))
    tfms.append(v2.ToImage())
    tfms.append(v2.ToDtype(torch.float32, scale=True))   # image 0..1 float
    return v2.Compose(tfms)


def detection_collate(batch):
    """RetinaNet expects (list_of_images, list_of_targets)."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def build_dataloaders(
    root,
    img_size=416,
    batch_size=8,
    num_workers=4,
    download=True,
):
    train_set = VOCDetectionAdapter(
        root, image_set="train",
        transforms=build_transforms(img_size, train=True),
        download=download,
    )
    val_set = VOCDetectionAdapter(
        root, image_set="val",
        transforms=build_transforms(img_size, train=False),
        download=download,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=detection_collate, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=detection_collate, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # M3 smoke test. Will download VOC on first run (~3 GB).
    train_loader, val_loader = build_dataloaders(
        root=PROJECT_ROOT / "data" / "voc",
        img_size=416, batch_size=4, num_workers=0, download=True,
    )
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    imgs, targets = next(iter(train_loader))
    print(f"batch size: {len(imgs)}")
    print(f"img[0] shape: {imgs[0].shape}, dtype={imgs[0].dtype}, range=[{imgs[0].min():.2f}, {imgs[0].max():.2f}]")
    print(f"target[0]: boxes {tuple(targets[0]['boxes'].shape)}  "
          f"labels {targets[0]['labels'].tolist()}")
    print("first box coords:", targets[0]['boxes'][0].tolist() if len(targets[0]['boxes']) else "no objects")
