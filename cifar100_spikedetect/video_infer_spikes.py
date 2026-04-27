"""Video detection + per-layer spike visualization for V3 detector.

Runs the same detection pipeline as video_infer.py, but additionally hooks
into the spiking backbone to capture per-layer spike tensors at each
timestep, then renders them as heatmap panels alongside the detection
overlay.

Layout:
  +-----------------------+------------------+
  |                       |  stem  | layer1  |
  |   Frame + boxes       +--------+---------+
  |                       |  layer2| layer3  |
  +-----------------------+--------+---------+
  |  fire-rate bar + top-K predictions       |
  +-------------------------------------------+

The spiking portion is the backbone only (SEW-ResNet-18 blocks). FPN and
RetinaNet head are float, so there are no spikes to visualize there.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms

from cifar100_spikedetect.data import VOC_CLASSES
from cifar100_spikedetect.model import build_retinanet


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--checkpoint", default="cifar100_spikedetect_V3_imagenet")
parser.add_argument("--score-thresh", type=float, default=0.25)
parser.add_argument("--class-agnostic-nms", type=float, default=0.45)
parser.add_argument("--top-k-per-frame", type=int, default=8)
parser.add_argument("--max-frames", type=int, default=200)
parser.add_argument("--frame-stride", type=int, default=3)
parser.add_argument("--img-size", type=int, default=416)
parser.add_argument("--device", default=None)
args = parser.parse_args()


# --- Model --------------------------------------------------------------

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck_path = PROJECT_ROOT / "checkpoints" / f"{args.checkpoint}_best.pth"
ckpt = torch.load(ck_path, map_location=device, weights_only=False)
classes = ckpt.get("classes", VOC_CLASSES)
img_size = ckpt.get("img_size", args.img_size)

model = build_retinanet(
    num_classes=len(classes),
    num_steps=ckpt.get("num_steps", 8),
    trainable_backbone_layers=3,
    min_size=img_size, max_size=img_size,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
model.score_thresh = args.score_thresh

print(f"Loaded {args.checkpoint} epoch {ckpt['epoch']} mAP@50={ckpt.get('map', 0):.3f}")


# --- Video I/O ----------------------------------------------------------

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W_src}x{H_src} @ {fps:.1f}fps, {total} frames")

# Output layout: left = frame (scaled to 640x360), right = 2x2 grid of spike panels
# Bottom strip: fire rates + predictions
MAIN_W, MAIN_H = 640, 360
PANEL_W, PANEL_H = 200, 180
SIDE_W = PANEL_W * 2 + 10
FOOTER_H = 90
OUT_W = MAIN_W + SIDE_W + 20
OUT_H = max(MAIN_H, PANEL_H * 2 + 10) + FOOTER_H

out_path = Path(args.out) if args.out else (
    PROJECT_ROOT / "viz" / f"{Path(args.video).stem}_spikes.mp4"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))


# --- Helpers ------------------------------------------------------------

def put_text(img, text, pos, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def heatmap_panel(spk_tensor_tbc_hw, panel_w, panel_h):
    """spk_tensor: [T, 1, C, H, W] (single-batch). Returns BGR uint8 panel."""
    # Sum spikes over T and C → [H, W] spike-count map
    smap = spk_tensor_tbc_hw[:, 0].sum(dim=(0, 1)).cpu().numpy()
    if smap.max() > 0:
        smap = np.clip(smap / smap.max() * 255, 0, 255).astype(np.uint8)
    else:
        smap = np.zeros_like(smap, dtype=np.uint8)
    return cv2.resize(cv2.applyColorMap(smap, cv2.COLORMAP_HOT),
                      (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)


rng = np.random.RandomState(42)
CLASS_COLORS = [(0, 0, 0)] + [tuple(int(c) for c in rng.randint(50, 230, 3)) for _ in classes[1:]]


_spike_records_holder = {}


def _install_spike_hook(model):
    """Monkey-patch backbone body forward to capture spikes each call."""
    body = model.backbone.body
    orig_forward = body.forward

    def wrapped(x):
        feats, records = orig_forward(x, record_spikes=True)
        _spike_records_holder['current'] = records
        return feats

    body.forward = wrapped


_install_spike_hook(model)


@torch.no_grad()
def run_inference(frame_bgr):
    """Run detector AND capture per-layer spikes from the backbone."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(img_size, img_size),
        mode="bilinear", align_corners=False,
    )[0].to(device)

    _spike_records_holder.clear()
    outs = model([t])
    out = outs[0]
    spike_records = _spike_records_holder.get('current', {})

    boxes = out["boxes"].cpu()
    labels = out["labels"].cpu()
    scores = out["scores"].cpu()
    keep = scores >= args.score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if args.class_agnostic_nms > 0 and len(boxes) > 0:
        keep_idx = tv_nms(boxes, scores, args.class_agnostic_nms)
        boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]
    if args.top_k_per_frame is not None and len(boxes) > args.top_k_per_frame:
        top = scores.argsort(descending=True)[: args.top_k_per_frame]
        boxes, labels, scores = boxes[top], labels[top], scores[top]

    sx = W_src / img_size
    sy = H_src / img_size
    dets = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        dets.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy,
                     int(label.item()), float(score.item())))

    # Compute mean fire rate per layer
    fire_rates = {k: float(v.float().mean().item()) for k, v in spike_records.items()}
    return dets, spike_records, fire_rates


def render_frame(frame_bgr, dets, spike_records, fire_rates):
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

    # Main panel: scaled frame + overlay boxes
    main = cv2.resize(frame_bgr, (MAIN_W, MAIN_H))
    scale_x, scale_y = MAIN_W / W_src, MAIN_H / H_src
    for x1, y1, x2, y2, label, score in dets:
        col = CLASS_COLORS[label]
        cv2.rectangle(main,
                      (int(x1 * scale_x), int(y1 * scale_y)),
                      (int(x2 * scale_x), int(y2 * scale_y)),
                      col, 2)
        put_text(main, f"{classes[label]} {score*100:.0f}%",
                 (int(x1 * scale_x) + 3, int(y1 * scale_y) + 16),
                 color=col, scale=0.45, thick=1)
    canvas[0:MAIN_H, 0:MAIN_W] = main

    # Spike panels (2x2 grid on the right)
    panel_layout = [
        ('stem', 0, 0), ('layer1', 0, 1),
        ('layer2', 1, 0), ('layer3', 1, 1),
    ]
    spike_origin_x = MAIN_W + 10
    for name, row, col in panel_layout:
        if name not in spike_records:
            continue
        x = spike_origin_x + col * (PANEL_W + 10)
        y = row * (PANEL_H + 10)
        panel = heatmap_panel(spike_records[name], PANEL_W, PANEL_H)
        canvas[y:y + PANEL_H, x:x + PANEL_W] = panel
        put_text(canvas, f"{name} fr={fire_rates.get(name, 0)*100:.1f}%",
                 (x + 4, y + 14), color=(255, 255, 255), scale=0.4, thick=1)

    # Footer: fire rates + top-3 preds
    fy = MAIN_H + 20
    put_text(canvas, f"SEW-RetinaNet V3 (ImageNet backbone) @ {img_size}px",
             (10, fy), scale=0.5, thick=1)
    fire_str = "  ".join(f"{k}={v*100:.1f}%" for k, v in fire_rates.items())
    put_text(canvas, f"Spike fire rates: {fire_str}",
             (10, fy + 22), scale=0.45, thick=1, color=(200, 200, 200))
    # top-3 detections by confidence
    if dets:
        top3 = sorted(dets, key=lambda d: -d[5])[:3]
        det_str = "  ".join(f"{classes[d[4]]}:{d[5]*100:.0f}%" for d in top3)
        put_text(canvas, f"Top detections: {det_str}",
                 (10, fy + 44), scale=0.45, thick=1, color=(0, 255, 255))
    return canvas


# --- Main loop ----------------------------------------------------------

frame_idx = 0
last_dets, last_spikes, last_fr = [], {}, {}
t0 = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    if args.max_frames is not None and frame_idx >= args.max_frames:
        break

    if frame_idx % args.frame_stride == 0:
        last_dets, last_spikes, last_fr = run_inference(frame_bgr)

    out_frame = render_frame(frame_bgr, last_dets, last_spikes, last_fr)
    writer.write(out_frame)
    frame_idx += 1

    if frame_idx % 25 == 0:
        elapsed = time.time() - t0
        fps_eff = frame_idx / elapsed
        print(f"  {frame_idx}/{total}  ({fps_eff:.2f} fps eff)  n_dets={len(last_dets)}  fr={last_fr}")

cap.release()
writer.release()
print(f"Saved: {out_path}")
