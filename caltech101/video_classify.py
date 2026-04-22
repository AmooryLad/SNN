"""Stream neuromorphic video through the N-Caltech101 SNN classifier.

Two input modes:
  --bin path/to/sample.bin    : real N-Caltech101 event stream (in-distribution)
  --video path/to/clip.mp4    : RGB video → pseudo-events via frame differencing
                                 (out-of-distribution — honest "deployment" test)

Output: annotated MP4 with side-by-side layout:
  left  = RGB frame (video mode) or accumulated events (bin mode)
  right = current event window (ON=red, OFF=blue)
  overlay = top-K class predictions

Example:
  python caltech101/video_classify.py --video /tmp/butterfly.mp4 --out viz/butterfly_events.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from caltech101.model import SNNConvClassifier
from caltech101.data import load_event_file


# --- CLI ------------------------------------------------------

parser = argparse.ArgumentParser()
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument('--bin', type=str, help='N-Caltech101 .bin event file')
src.add_argument('--video', type=str, help='RGB video (mp4); will be frame-diffed to events')
parser.add_argument('--out', default=None)
parser.add_argument('--checkpoint', default='caltech101_phase3_v2')
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--window-ms', type=float, default=200.0,
                    help='events per prediction window (ms)')
parser.add_argument('--stride-ms', type=float, default=80.0,
                    help='how far to slide window between predictions (ms)')
parser.add_argument('--event-thresh', type=float, default=0.12,
                    help='log-intensity threshold for frame-diff event gen')
parser.add_argument('--max-frames', type=int, default=None)
args = parser.parse_args()


# --- Model ----------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = PROJECT_ROOT / 'checkpoints' / f'{args.checkpoint}_best.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
classes = ckpt['classes']
num_steps = ckpt['num_steps']
depth = ckpt['depth']
bn_mode = ckpt['bn_mode']

net = SNNConvClassifier(
    num_classes=len(classes), num_steps=num_steps,
    depth=depth, bn_mode=bn_mode,
).to(device)
net.load_state_dict(ckpt['model_state_dict'])
net.eval()
print(f"Loaded {args.checkpoint} epoch {ckpt['epoch']} "
      f"(acc {ckpt['accuracy']:.2f}%)  T={num_steps}, depth={depth}")

H, W = 180, 240  # N-Caltech101 input size


@torch.no_grad()
def classify_window(frames_tbhw):
    """frames_tbhw: [T, 2, H, W] numpy float32.
    Returns (probs [num_classes], fire_rates [list], records [dict of [T,B,C,H,W]]).
    """
    x = torch.from_numpy(frames_tbhw).unsqueeze(1).to(device)
    spk_out_rec, mem_out, fire_rates, records = net(x, record_all=True)
    probs = torch.softmax(mem_out, dim=1)[0].cpu()
    fr = [float(f.item()) for f in fire_rates]
    # Sum each record over T and C → [H, W] spatial spike density
    heatmaps = {}
    for name in ('lif1', 'lif2', 'lif3', 'lif4'):
        if name in records:
            rec = records[name][:, 0]  # [T, C, H, W]
            hmap = rec.sum(dim=(0, 1)).cpu().numpy()
            heatmaps[name] = hmap
    return probs, fr, heatmaps


# --- Event source --------------------------------------------

def bin_events_to_frames(x, y, p, t, n_steps, t_start, t_end, H, W):
    """Bin events in [t_start, t_end] into n_steps frames [n_steps, 2, H, W]."""
    frames = np.zeros((n_steps, 2, H, W), dtype=np.float32)
    if len(t) == 0 or t_end <= t_start:
        return frames
    mask = (t >= t_start) & (t < t_end)
    if not mask.any():
        return frames
    xw, yw, pw, tw = x[mask], y[mask], p[mask], t[mask]
    bins = ((tw - t_start) / (t_end - t_start) * (n_steps - 1)).astype(int)
    bins = np.clip(bins, 0, n_steps - 1)
    xw = np.clip(xw, 0, W - 1)
    yw = np.clip(yw, 0, H - 1)
    pw = np.clip(pw, 0, 1)
    np.add.at(frames, (bins, pw, yw, xw), 1.0)
    return frames


def events_from_bin(path):
    """Load entire .bin as (x, y, p, t) arrays in microseconds."""
    return load_event_file(path)


def events_from_video(path, thresh=0.12, max_frames=None):
    """Generate pseudo-events by frame differencing an RGB video.

    Returns (x, y, p, t) arrays with t in microseconds.
    Also returns the source frame sequence so we can visualize alongside.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"RGB source: {src_W}x{src_H} @ {fps:.1f}fps")

    xs, ys, ps, ts = [], [], [], []
    src_frames = []
    prev_log = None
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        bgr_small = cv2.resize(bgr, (W, H))
        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        log_img = np.log(gray + 1.0)

        if prev_log is not None:
            diff = log_img - prev_log
            on_mask = diff > thresh
            off_mask = diff < -thresh
            ty, tx = np.where(on_mask)
            for yi, xi in zip(ty, tx):
                xs.append(xi); ys.append(yi); ps.append(1)
                ts.append(int(frame_idx * 1e6 / fps))
            ty, tx = np.where(off_mask)
            for yi, xi in zip(ty, tx):
                xs.append(xi); ys.append(yi); ps.append(0)
                ts.append(int(frame_idx * 1e6 / fps))

        prev_log = log_img
        src_frames.append(bgr)
        frame_idx += 1

    cap.release()
    print(f"Generated {len(xs)} events from {frame_idx} frames")
    return (np.array(xs, np.int32),
            np.array(ys, np.int32),
            np.array(ps, np.int32),
            np.array(ts, np.int64),
            src_frames, fps)


if args.bin is not None:
    x, y, p, t = events_from_bin(args.bin)
    src_frames, src_fps = [], 30.0
    total_time_us = int(t.max() - t.min())
    mode_label = f"N-Caltech .bin: {Path(args.bin).parent.name}/{Path(args.bin).name}"
else:
    x, y, p, t, src_frames, src_fps = events_from_video(
        args.video, thresh=args.event_thresh, max_frames=args.max_frames,
    )
    total_time_us = int(t.max() - t.min()) if len(t) else 0
    mode_label = f"pseudo-events from {Path(args.video).name}"

if total_time_us == 0:
    print("No events generated. Check --event-thresh.")
    sys.exit(1)

t_min = int(t.min())
print(f"Event stream: {len(t)} events over {total_time_us/1e6:.2f}s")


# --- Output video --------------------------------------------

# Layout:
#  header 60 | main row 180 (source | events) | spike row 160 (4 layer panels) | footer for predictions 60
LAYER_PANEL_W, LAYER_PANEL_H = 120, 100
SPIKE_ROW_Y = 80 + H + 20     # top-of-spike-row on the canvas
out_W = max(W * 2 + 20, LAYER_PANEL_W * 4 + 20 + 30)
out_H = SPIKE_ROW_Y + LAYER_PANEL_H + 80  # + footer
out_path = Path(args.out) if args.out else (
    PROJECT_ROOT / 'viz' / f'{Path(args.bin or args.video).stem}_ncaltech.mp4'
)
out_path.parent.mkdir(parents=True, exist_ok=True)

out_fps = 30.0 if args.bin else src_fps
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_W, out_H))
print(f"Writing -> {out_path}  ({out_W}x{out_H} @ {out_fps:.1f}fps)")


def heatmap_to_rgb(hmap, panel_w, panel_h, cmap=cv2.COLORMAP_HOT):
    """[H,W] spike-count heatmap -> BGR panel resized to panel_w x panel_h."""
    if hmap.max() <= 0:
        gray = np.zeros_like(hmap, dtype=np.uint8)
    else:
        gray = np.clip(hmap / hmap.max() * 255, 0, 255).astype(np.uint8)
    panel = cv2.applyColorMap(gray, cmap)
    return cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)


def event_frame_to_rgb(ev_frame_thw):
    """ev_frame_thw: [T, 2, H, W]. Returns H, W, 3 BGR uint8."""
    on = ev_frame_thw[:, 1].sum(axis=0)
    off = ev_frame_thw[:, 0].sum(axis=0)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    scale = max(1.0, max(on.max(), off.max()))
    canvas[..., 2] = np.clip(on / scale * 255, 0, 255)   # red = ON
    canvas[..., 0] = np.clip(off / scale * 255, 0, 255)  # blue = OFF
    return canvas.astype(np.uint8)


def put_text(img, text, pos, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


# --- Streaming loop ------------------------------------------

window_us = int(args.window_ms * 1000)
stride_us = int(args.stride_ms * 1000)

cur = t_min
pred_idx = 0
last_probs = None

t0 = time.time()
# Emit one output frame per stride step, which becomes one "video frame"
while cur + window_us <= t_min + total_time_us:
    ev_frame = bin_events_to_frames(
        x, y, p, t, num_steps, cur, cur + window_us, H, W,
    )
    probs, fire_rates, heatmaps = classify_window(ev_frame)
    top_vals, top_idx = torch.topk(probs, args.top_k)
    last_probs = (top_vals.tolist(), top_idx.tolist())

    # Compose output canvas
    canvas = np.zeros((out_H, out_W, 3), dtype=np.uint8)

    # Left panel: source frame (or event accumulation)
    if src_frames:
        src_idx = int((cur - t_min) / total_time_us * len(src_frames))
        src_idx = min(src_idx, len(src_frames) - 1)
        left = cv2.resize(src_frames[src_idx], (W, H))
    else:
        left = event_frame_to_rgb(ev_frame)
    canvas[80:80 + H, 0:W] = left

    # Right panel: current event window
    right = event_frame_to_rgb(ev_frame)
    canvas[80:80 + H, W + 20:2 * W + 20] = right

    # Header
    put_text(canvas, mode_label, (10, 22), scale=0.55)
    put_text(canvas,
             f'window: [{(cur - t_min) / 1000:.0f}ms - {(cur - t_min + window_us) / 1000:.0f}ms]',
             (10, 45), scale=0.45, color=(180, 180, 180))
    put_text(canvas, 'source / events', (W - 80, 72), scale=0.45,
             color=(200, 200, 200))
    put_text(canvas, 'event window (R=ON, B=OFF)', (W + 25, 72), scale=0.45,
             color=(200, 200, 200))

    # Spike row: one heatmap per layer + fire rate label
    fire_labels = [f"lif{i+1}" for i in range(depth)]
    for i, lname in enumerate(fire_labels):
        if lname not in heatmaps:
            continue
        x0 = 10 + i * (LAYER_PANEL_W + 10)
        panel = heatmap_to_rgb(heatmaps[lname], LAYER_PANEL_W, LAYER_PANEL_H)
        canvas[SPIKE_ROW_Y:SPIKE_ROW_Y + LAYER_PANEL_H,
               x0:x0 + LAYER_PANEL_W] = panel
        cv2.rectangle(canvas,
                      (x0 - 1, SPIKE_ROW_Y - 1),
                      (x0 + LAYER_PANEL_W + 1, SPIKE_ROW_Y + LAYER_PANEL_H + 1),
                      (100, 100, 100), 1)
        put_text(canvas,
                 f'{lname}  fr={fire_rates[i]*100:.1f}%',
                 (x0 + 4, SPIKE_ROW_Y - 6),
                 scale=0.4, thick=1, color=(255, 255, 255))

    # Predictions — in footer row below spike panels
    if last_probs is not None:
        tv, ti = last_probs
        footer_y = SPIKE_ROW_Y + LAYER_PANEL_H + 22
        for v, i in zip(tv, ti):
            put_text(canvas,
                     f'{classes[i]:<20s} {v*100:5.1f}%',
                     (10, footer_y), color=(0, 255, 255), scale=0.55, thick=1)
            footer_y += 22

    writer.write(canvas)
    cur += stride_us
    pred_idx += 1

    if pred_idx % 25 == 0:
        print(f"  {pred_idx} predictions  ({pred_idx / (time.time() - t0):.1f} fps)")

writer.release()
print(f"Saved: {out_path}  ({pred_idx} classification windows)")
