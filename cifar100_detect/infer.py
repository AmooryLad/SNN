"""Single-image inference + visualization for the SEW RetinaNet detector."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from cifar100_detect.data import VOC_CLASSES
from cifar100_detect.model import build_retinanet


parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--checkpoint", default="cifar100_detect_V1")
parser.add_argument("--out", default=None)
parser.add_argument("--score-thresh", type=float, default=0.3)
parser.add_argument("--img-size", type=int, default=416)
args = parser.parse_args()


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
print(f"Loaded {args.checkpoint} epoch {ckpt['epoch']}  mAP@50={ckpt.get('map', 0):.3f}")


img_pil = Image.open(args.image).convert("RGB")
W0, H0 = img_pil.size
img_np = np.array(img_pil)

tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
tensor = torch.nn.functional.interpolate(
    tensor.unsqueeze(0), size=(img_size, img_size),
    mode="bilinear", align_corners=False,
)[0].to(device)

with torch.no_grad():
    out = model([tensor])[0]

sx, sy = W0 / img_size, H0 / img_size
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

rng = np.random.RandomState(42)
colors = [(0, 0, 0)] + [tuple(int(c) for c in rng.randint(50, 230, size=3)) for _ in classes[1:]]

for box, label, score in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
    if score.item() < args.score_thresh:
        continue
    x1, y1, x2, y2 = box.tolist()
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    color = colors[int(label.item())]
    cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    tag = f"{classes[int(label.item())]} {score.item()*100:.0f}%"
    cv2.putText(img_bgr, tag, (int(x1) + 4, int(y1) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_bgr, tag, (int(x1) + 4, int(y1) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

out_path = Path(args.out) if args.out else (
    PROJECT_ROOT / "viz" / f"{Path(args.image).stem}_detected.jpg"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), img_bgr)
print(f"Detections: {(out['scores'] >= args.score_thresh).sum().item()}")
print(f"Saved: {out_path}")
