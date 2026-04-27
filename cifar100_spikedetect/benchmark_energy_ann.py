"""ANN baseline energy benchmark for direct apples-to-apples comparison
with our hybrid SNN-ANN detector.

Same video, same CPU, same thread count, same energy methodology.
Loads a pretrained torchvision detector (default: Faster R-CNN MobileNetV3
which is ~19M params — the closest match to our 20M).

Run:
  python cifar100_spikedetect/benchmark_energy_ann.py --video /tmp/dashcam.webm \
      --max-frames 300 --frame-stride 3 --num-threads 16
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn",
                    choices=["fasterrcnn_mobilenet_v3_large_fpn",
                             "fasterrcnn_mobilenet_v3_large_320_fpn",
                             "ssdlite320_mobilenet_v3_large",
                             "retinanet_resnet50_fpn_v2",
                             "fcos_resnet50_fpn"],
                    help="torchvision pretrained detector to benchmark")
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--frame-stride", type=int, default=1)
parser.add_argument("--score-thresh", type=float, default=0.25)
parser.add_argument("--img-size", type=int, default=416)
parser.add_argument("--num-threads", type=int, default=None)
parser.add_argument("--cpu-tdp", type=float, default=200.0)
parser.add_argument("--no-warmup", action="store_true")
args = parser.parse_args()


# --- Setup --------------------------------------------------------------

torch.set_num_threads(args.num_threads or os.cpu_count())
device = torch.device("cpu")
print(f"Device: CPU  Threads: {torch.get_num_threads()}")
print(f"Logical CPUs: {os.cpu_count()}  ")
print(f"Estimated power per active thread: {args.cpu_tdp / os.cpu_count():.2f} W")


# --- Load ANN detector --------------------------------------------------

print(f"\nLoading ANN baseline: {args.model}")
import torchvision.models.detection as tvd

if args.model == "fasterrcnn_mobilenet_v3_large_fpn":
    model = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT",
                                                    min_size=args.img_size,
                                                    max_size=args.img_size)
elif args.model == "fasterrcnn_mobilenet_v3_large_320_fpn":
    model = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
elif args.model == "ssdlite320_mobilenet_v3_large":
    model = tvd.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
elif args.model == "retinanet_resnet50_fpn_v2":
    model = tvd.retinanet_resnet50_fpn_v2(weights="DEFAULT",
                                           min_size=args.img_size,
                                           max_size=args.img_size)
elif args.model == "fcos_resnet50_fpn":
    model = tvd.fcos_resnet50_fpn(weights="DEFAULT",
                                    min_size=args.img_size,
                                    max_size=args.img_size)
else:
    raise ValueError(f"Unknown model: {args.model}")

model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params:,}  ({n_params * 4 / 1024**2:.1f} MB fp32)")


# --- Open video ---------------------------------------------------------

cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nVideo: {args.video}")
print(f"  {W_src}x{H_src} @ {fps_in:.1f} fps, {total} frames "
      f"({total / fps_in:.1f}s duration)")
print(f"  Frame stride: {args.frame_stride}")


# --- Warmup -------------------------------------------------------------

if not args.no_warmup:
    print("\nWarming up (5 dummy forward passes)...")
    dummy = torch.randn(3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model([dummy])
    print("  Warmup done.")


# --- Benchmark loop -----------------------------------------------------

print(f"\n=== Starting ANN benchmark ===")
proc = psutil.Process()
cpu_times_start = proc.cpu_times()
mem_start = proc.memory_info().rss / 1024**2

wall_start = time.time()
inference_total = 0.0
preprocess_total = 0.0
total_dets = 0
frame_idx = 0
processed = 0
per_frame_latencies = []

with torch.no_grad():
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames is not None and processed >= args.max_frames:
            break
        if frame_idx % args.frame_stride != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(args.img_size, args.img_size),
            mode="bilinear", align_corners=False,
        )[0]
        t1 = time.perf_counter()
        preprocess_total += (t1 - t0)

        t2 = time.perf_counter()
        out = model([t])[0]
        t3 = time.perf_counter()
        inference_total += (t3 - t2)
        per_frame_latencies.append(t3 - t2)

        valid = (out["scores"] >= args.score_thresh).sum().item()
        total_dets += valid

        processed += 1
        frame_idx += 1

        if processed % 25 == 0:
            elapsed = time.time() - wall_start
            fps_eff = processed / elapsed
            print(f"  {processed} frames done | {fps_eff:.2f} fps eff | "
                  f"avg latency {np.mean(per_frame_latencies)*1000:.1f} ms | "
                  f"dets last frame: {valid}")

cap.release()
wall_end = time.time()
cpu_times_end = proc.cpu_times()
mem_end = proc.memory_info().rss / 1024**2


# --- Results ------------------------------------------------------------

wall_secs = wall_end - wall_start
process_cpu_secs = (
    (cpu_times_end.user - cpu_times_start.user) +
    (cpu_times_end.system - cpu_times_start.system)
)
power_per_thread = args.cpu_tdp / os.cpu_count()
estimated_energy_joules = process_cpu_secs * power_per_thread
energy_per_frame_mj = (estimated_energy_joules / processed) * 1000

print("\n" + "=" * 60)
print(f"=== ANN BASELINE RESULTS: {args.model} ===")
print("=" * 60)
print(f"\nFrames processed:         {processed}")
print(f"Total wall-clock time:    {wall_secs:.2f} s")
print(f"Pure inference time:      {inference_total:.2f} s")
print(f"Process CPU-seconds:      {process_cpu_secs:.2f} s")
print(f"Memory delta:             {mem_end - mem_start:+.1f} MB")
print(f"\n--- Latency ---")
print(f"  Mean per-frame:         {np.mean(per_frame_latencies)*1000:.1f} ms")
print(f"  Median per-frame:       {np.median(per_frame_latencies)*1000:.1f} ms")
print(f"  P95 per-frame:          {np.percentile(per_frame_latencies, 95)*1000:.1f} ms")
print(f"  Inference-only FPS:     {processed / inference_total:.2f}")
print(f"  End-to-end FPS (wall):  {processed / wall_secs:.2f}")
print(f"\n--- Energy estimate ---")
print(f"  CPU-seconds total:      {process_cpu_secs:.2f}")
print(f"  Estimated energy:       {estimated_energy_joules:.2f} J")
print(f"  Energy per frame:       {energy_per_frame_mj:.2f} mJ")
print(f"\n--- Detection summary ---")
print(f"  Total detections (above {args.score_thresh}): {total_dets}")
print(f"  Average per frame:      {total_dets / processed:.2f}")
print("\n" + "=" * 60)
