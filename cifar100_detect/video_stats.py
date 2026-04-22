"""Print per-frame detections and a class-frequency summary for a video."""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import torch

from cifar100_detect.data import VOC_CLASSES
from cifar100_detect.model import build_retinanet


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--checkpoint", default="cifar100_detect_V2_coco")
parser.add_argument("--score-thresh", type=float, default=0.15)
parser.add_argument("--frame-stride", type=int, default=10)
parser.add_argument("--max-frames", type=int, default=None)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(PROJECT_ROOT / "checkpoints" / f"{args.checkpoint}_best.pth",
                map_location=device, weights_only=False)
classes = ck.get("classes", VOC_CLASSES)
img_size = ck.get("img_size", 416)

model = build_retinanet(
    num_classes=len(classes),
    num_steps=ck.get("num_steps", 8),
    trainable_backbone_layers=3,
    min_size=img_size, max_size=img_size,
)
model.load_state_dict(ck["model_state_dict"])
model.to(device).eval()
model.score_thresh = args.score_thresh
print(f"Model: {args.checkpoint}  epoch {ck['epoch']}  mAP@50={ck.get('map', 0):.3f}")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {args.video}  {total} frames @ {fps:.1f}fps")
print(f"Sampling every {args.frame_stride}th frame at score >= {args.score_thresh}\n")

freq = Counter()
frame_idx = 0
sampled = 0

with torch.no_grad():
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        if frame_idx % args.frame_stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(img_size, img_size),
                mode="bilinear", align_corners=False,
            )[0].to(device)
            out = model([t])[0]
            dets = [(int(l.item()), float(s.item()))
                    for l, s in zip(out["labels"].cpu(), out["scores"].cpu())
                    if s.item() >= args.score_thresh]
            secs = frame_idx / fps
            if dets:
                summary = " | ".join(
                    f"{classes[l]}:{s*100:.0f}%" for l, s in dets[:5]
                )
                print(f"  t={secs:5.2f}s  frame={frame_idx:5d}  ->  {summary}")
            else:
                print(f"  t={secs:5.2f}s  frame={frame_idx:5d}  ->  (no detections)")
            for l, _ in dets:
                freq[classes[l]] += 1
            sampled += 1
        frame_idx += 1

cap.release()
print(f"\n=== Summary over {sampled} sampled frames ===")
for cls, n in freq.most_common():
    print(f"  {cls:20s}  {n} detections")
