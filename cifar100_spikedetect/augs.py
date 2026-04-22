"""Mosaic and MixUp augmentation for detection datasets.

Wraps a standard (image, target) detection dataset and probabilistically
returns either:
  - a vanilla sample (1-prob of mosaic)
  - a 2x2 mosaic of 4 random samples (prob of mosaic)

MixUp is applied batch-wise inside the training loop (see mixup_batch).

Bounding boxes are tracked through every transform:
  - Mosaic: each of the 4 sub-image's boxes are translated + scaled into
    the big mosaic canvas, then clipped to canvas bounds, then the mosaic
    is resized back to img_size.
  - MixUp: two image tensors are convex-combined; label lists are
    concatenated (classic detection MixUp).

Final filter: drop any box smaller than 4 pixels in either dim (prevents
degenerate regression targets).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --- Helpers ----------------------------------------------------------------

def _paste(canvas, patch, x, y):
    """Paste patch onto canvas at top-left (x, y), clipping to canvas size."""
    _, ch, cw = canvas.shape
    _, ph, pw = patch.shape
    dx1, dy1 = max(x, 0), max(y, 0)
    dx2, dy2 = min(x + pw, cw), min(y + ph, ch)
    sx1, sy1 = dx1 - x, dy1 - y
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
    canvas[:, dy1:dy2, dx1:dx2] = patch[:, sy1:sy2, sx1:sx2]


def _translate_boxes(boxes, dx, dy, W, H, scale_x=1.0, scale_y=1.0):
    """Translate + scale boxes, clip to (0,0,W,H)."""
    b = boxes.clone().float()
    b[:, [0, 2]] = b[:, [0, 2]] * scale_x + dx
    b[:, [1, 3]] = b[:, [1, 3]] * scale_y + dy
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
    return b


def _filter_tiny(boxes, labels, min_side=4):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_side) & (h >= min_side)
    return boxes[keep], labels[keep]


# --- Mosaic wrapper -------------------------------------------------------

class MosaicDetection(Dataset):
    """Wrap a detection dataset to probabilistically return a 4-image mosaic.

    Expects base[i] to return (image [C,H,W], target dict with 'boxes', 'labels').
    Produces the same format.
    """

    def __init__(self, base, img_size=416, prob=0.5, pivot_range=(0.35, 0.65)):
        self.base = base
        self.img_size = img_size
        self.prob = prob
        self.pivot_range = pivot_range

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if random.random() > self.prob:
            return self.base[idx]
        return self._mosaic(idx)

    def _mosaic(self, idx):
        size = self.img_size
        pivot_x = int(size * random.uniform(*self.pivot_range))
        pivot_y = int(size * random.uniform(*self.pivot_range))

        # Pick 4 images: current + 3 others
        indices = [idx] + random.sample(range(len(self.base)), 3)
        slot_coords = [
            (0, 0, pivot_x, pivot_y),            # top-left slot
            (pivot_x, 0, size - pivot_x, pivot_y),  # top-right
            (0, pivot_y, pivot_x, size - pivot_y),  # bottom-left
            (pivot_x, pivot_y, size - pivot_x, size - pivot_y),  # bottom-right
        ]

        canvas = torch.zeros(3, size, size, dtype=torch.float32)
        all_boxes, all_labels = [], []

        for i, (ix, iy, iw, ih) in enumerate(slot_coords):
            img_i, tgt_i = self.base[indices[i]]
            # If img is already HxW tensor at img_size, resize to slot dims.
            _, src_h, src_w = img_i.shape
            # Resize image into this slot
            slot = F.interpolate(img_i.unsqueeze(0),
                                 size=(ih, iw),
                                 mode='bilinear', align_corners=False)[0]
            canvas[:, iy:iy + ih, ix:ix + iw] = slot

            # Scale + translate boxes
            if len(tgt_i["boxes"]) > 0:
                scale_x = iw / src_w
                scale_y = ih / src_h
                bxs = _translate_boxes(tgt_i["boxes"], ix, iy, size, size,
                                       scale_x=scale_x, scale_y=scale_y)
                all_boxes.append(bxs)
                all_labels.append(tgt_i["labels"])

        boxes = (torch.cat(all_boxes, dim=0) if all_boxes
                 else torch.zeros((0, 4), dtype=torch.float32))
        labels = (torch.cat(all_labels, dim=0) if all_labels
                  else torch.zeros((0,), dtype=torch.int64))
        boxes, labels = _filter_tiny(boxes, labels, min_side=4)
        return canvas, {"boxes": boxes, "labels": labels}


# --- MixUp ---------------------------------------------------------------

def mixup_batch(imgs, targets, prob=0.5, alpha=8.0):
    """Apply MixUp to the whole batch with probability `prob`.

    Pairs image i with image ((i+1) % B). Creates a convex combination of the
    pixels; concatenates the box+label lists. RetinaNet's loss happily accepts
    any number of boxes per image, so this is straightforward.
    """
    if random.random() > prob or len(imgs) < 2:
        return imgs, targets

    B = len(imgs)
    lam = float(np.random.beta(alpha, alpha))
    new_imgs, new_targets = [], []
    for i in range(B):
        j = (i + 1) % B
        img_a, img_b = imgs[i], imgs[j]
        tgt_a, tgt_b = targets[i], targets[j]
        # Ensure same HxW (all images are already 416x416 in our pipeline)
        mixed = lam * img_a + (1.0 - lam) * img_b
        new_imgs.append(mixed)
        new_targets.append({
            "boxes": torch.cat([tgt_a["boxes"], tgt_b["boxes"]], dim=0),
            "labels": torch.cat([tgt_a["labels"], tgt_b["labels"]], dim=0),
        })
    return new_imgs, new_targets
