"""Run the CIFAR-100 SEW-ResNet classifier on each frame of a video.

The model is a 32x32 classifier, not a detector — this writes an annotated
output video with per-frame top-K predictions. Two modes:

  --mode center   : center-crop + resize each frame, one classification.
  --mode sliding  : NxN grid of crops, classify each, produce class heatmap
                    overlaid on the frame (OverFeat-style proto-detection).

Usage:
  python cifar100_sewresnet/video_classify.py --video path.mp4 \
      --mode center --out viz/test_out.mp4

A video URL can also be passed; it will be downloaded to /tmp first.
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from cifar100_sewresnet.model import SEWResNet18
from cifar100_sewresnet.data import CIFAR100_MEAN, CIFAR100_STD


# --- CLI -----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True,
                    help='path to local .mp4 or a URL to download')
parser.add_argument('--out', default=None,
                    help='output annotated mp4 (default: viz/<name>_annotated.mp4)')
parser.add_argument('--mode', choices=['center', 'sliding'], default='center')
parser.add_argument('--checkpoint', default='cifar100_sew_T',
                    help='checkpoint name under checkpoints/ (no suffix)')
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--frame-stride', type=int, default=1,
                    help='classify every Nth frame (fill others with prev pred)')
parser.add_argument('--max-frames', type=int, default=None,
                    help='cap total processed frames (debug)')
parser.add_argument('--grid', type=int, default=4,
                    help='grid size for sliding-window mode')
args = parser.parse_args()


# --- Video source --------------------------------------------------------

video_path = args.video
if video_path.startswith(('http://', 'https://')):
    cache = Path('/tmp') / Path(video_path).name
    if not cache.exists():
        print(f"Downloading {video_path} -> {cache}")
        urllib.request.urlretrieve(video_path, cache)
    video_path = str(cache)
    print(f"Using cached video: {video_path}")

if not os.path.exists(video_path):
    print(f"Error: video not found: {video_path}")
    sys.exit(1)


# --- Model ---------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = PROJECT_ROOT / 'checkpoints' / f'{args.checkpoint}_best.pth'
if not ckpt_path.exists():
    print(f"Error: checkpoint not found at {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=device)
classes = ckpt['classes']
num_steps = ckpt.get('num_steps', 8)

net = SEWResNet18(num_classes=len(classes), num_steps=num_steps).to(device)
# Prefer EMA weights if present
state = ckpt.get('ema_state_dict') or ckpt['model_state_dict']
net.load_state_dict(state)
net.eval()
print(f"Loaded {args.checkpoint} epoch {ckpt['epoch']} "
      f"(acc {ckpt['accuracy']:.2f}%)  steps={num_steps}")

normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)


def preprocess_tile(rgb_uint8):
    """HxWx3 uint8 RGB -> 1x3x32x32 normalized tensor on device."""
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False
    )[0]
    t = normalize(t).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def classify(tile_tensor):
    _, _, _, logit_rec = net(tile_tensor)
    logits = logit_rec.mean(dim=0)
    probs = torch.softmax(logits, dim=1)[0]
    return probs


# --- Video I/O ----------------------------------------------------------

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H} @ {fps:.1f}fps, {total} frames")

out_path = Path(args.out) if args.out else \
    PROJECT_ROOT / 'viz' / f'{Path(video_path).stem}_annotated.mp4'
out_path.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))


# --- Overlay helpers -----------------------------------------------------

def put_text(img, text, pos, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def annotate_center(frame_bgr, probs):
    """One set of top-K predictions on the frame."""
    top_vals, top_idx = torch.topk(probs, args.top_k)
    y = 30
    put_text(frame_bgr, f'CIFAR-100 SNN [center-crop mode]', (10, y))
    y += 30
    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
        put_text(frame_bgr,
                 f'{classes[i]:<20s} {v*100:5.1f}%',
                 (10, y), color=(0, 255, 255), scale=0.7)
        y += 28
    return frame_bgr


def annotate_sliding(frame_bgr, grid_probs, grid_size):
    """Overlay class labels on each cell of an NxN grid."""
    cell_h, cell_w = H // grid_size, W // grid_size
    for (gy, gx), probs in grid_probs.items():
        top1 = int(torch.argmax(probs).item())
        conf = float(probs[top1])
        if conf < 0.12:
            continue  # skip low-confidence cells
        x0, y0 = gx * cell_w, gy * cell_h
        x1, y1 = x0 + cell_w, y0 + cell_h
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 255), 1)
        put_text(frame_bgr,
                 f'{classes[top1]} {conf*100:.0f}%',
                 (x0 + 5, y0 + 22), color=(0, 255, 255), scale=0.5, thick=1)
    put_text(frame_bgr, f'CIFAR-100 SNN [{grid_size}x{grid_size} sliding]',
             (10, 30))
    return frame_bgr


# --- Main loop -----------------------------------------------------------

processed = 0
last_probs = None
last_grid_probs = None
t0 = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if args.max_frames is not None and processed >= args.max_frames:
        break

    run_model = (processed % args.frame_stride == 0)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if args.mode == 'center':
        if run_model:
            side = min(H, W)
            y0 = (H - side) // 2
            x0 = (W - side) // 2
            tile = frame_rgb[y0:y0 + side, x0:x0 + side]
            probs = classify(preprocess_tile(tile))
            last_probs = probs
        if last_probs is not None:
            annotate_center(frame_bgr, last_probs)

    else:  # sliding
        if run_model:
            g = args.grid
            cell_h, cell_w = H // g, W // g
            grid_probs = {}
            # Batch all grid tiles into one forward pass
            tiles = []
            keys = []
            for gy in range(g):
                for gx in range(g):
                    tile = frame_rgb[gy*cell_h:(gy+1)*cell_h,
                                     gx*cell_w:(gx+1)*cell_w]
                    tiles.append(preprocess_tile(tile))
                    keys.append((gy, gx))
            batch = torch.cat(tiles, dim=0)
            with torch.no_grad():
                _, _, _, logit_rec = net(batch)
                probs = torch.softmax(logit_rec.mean(dim=0), dim=1)
            for k, p in zip(keys, probs):
                grid_probs[k] = p.cpu()
            last_grid_probs = grid_probs
        if last_grid_probs is not None:
            annotate_sliding(frame_bgr, last_grid_probs, args.grid)

    writer.write(frame_bgr)
    processed += 1

    if processed % 50 == 0:
        elapsed = time.time() - t0
        fps_eff = processed / elapsed
        print(f"  {processed}/{total} frames  ({fps_eff:.1f} fps)")

cap.release()
writer.release()
print(f"Saved: {out_path}")
