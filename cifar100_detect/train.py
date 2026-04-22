"""Training loop for SEW-ResNet RetinaNet on Pascal VOC 2012 or COCO 2017.

Pattern mirrors cifar100_sewresnet/train.py: single-file config at top,
Tee logger, checkpoint on best mAP, cosine LR with linear warmup.

Differential fine-tuning: head/FPN get 10x higher LR than the spiking backbone.

V2_coco adds: AMP (mixed precision), larger batch, warm-start from VOC
checkpoint, and cudnn.benchmark for throughput.

Run:
    python cifar100_detect/train.py
"""

import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cifar100_detect.model import build_retinanet


# --- Experiment config ---------------------------------------------------

EXPERIMENT = "V2_coco"

EXPERIMENTS = {
    # name: (dataset, num_epochs, warmup_epochs, max_iters, do_eval, warm_start_ckpt, label)
    "V0_debug": ("voc",  1,  0, 50,   False, None,
                 "M4 smoke — 50 iters, no eval"),
    "V1":       ("voc",  30, 5, None, True,  None,
                 "M5 production run on VOC"),
    "V2_coco":  ("coco", 15, 2, None, True,  "cifar100_detect_V1_best.pth",
                 "M7 COCO warm-start from VOC + AMP + bigger batch"),
}

(dataset_name, num_epochs, warmup_epochs, max_iters,
 do_eval, warm_start_ckpt, exp_desc) = EXPERIMENTS[EXPERIMENT]
EXPERIMENT_NAME = f"cifar100_detect_{EXPERIMENT}"


# --- Hyperparameters -----------------------------------------------------

SEED = 42
img_size = 416
num_steps = 8

# Dataset-dependent settings.
if dataset_name == "coco":
    batch_size = 20            # AMP + 24 GB GPU headroom
    num_workers = 10
    lr_head = 3e-4             # slightly lower than VOC since dataset is much bigger
    lr_backbone = 3e-5
    use_amp = True
else:
    batch_size = 8
    num_workers = 4
    lr_head = 5e-4
    lr_backbone = 5e-5
    use_amp = False

weight_decay = 1e-4
grad_clip_norm = 1.0
trainable_backbone_layers = 3

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # faster once shapes stabilize


# --- Log tee -------------------------------------------------------------

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"{EXPERIMENT_NAME}.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"=== Experiment {EXPERIMENT}: {exp_desc} ===")
print(f"Log: {log_path}")
print(f"dataset={dataset_name}  batch={batch_size}  img={img_size}  "
      f"epochs={num_epochs}  warmup={warmup_epochs}  amp={use_amp}")


# --- Data ----------------------------------------------------------------

if dataset_name == "coco":
    from cifar100_detect.data_coco import build_dataloaders as build_coco_loaders
    train_loader, val_loader, class_names = build_coco_loaders(
        root=str(PROJECT_ROOT / "data" / "coco"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
    )
    num_classes = len(class_names)          # 81 (incl. background)
else:
    from cifar100_detect.data import build_dataloaders as build_voc_loaders, VOC_CLASSES
    train_loader, val_loader = build_voc_loaders(
        root=str(PROJECT_ROOT / "data" / "voc"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        download=False,
    )
    class_names = VOC_CLASSES
    num_classes = len(class_names)          # 21

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
print(f"num_classes={num_classes} (incl. background)")


# --- Model ---------------------------------------------------------------

base_backbone_ckpt = PROJECT_ROOT / "checkpoints" / "cifar100_sew_T_best.pth"
model = build_retinanet(
    num_classes=num_classes,
    num_steps=num_steps,
    backbone_ckpt=str(base_backbone_ckpt) if base_backbone_ckpt.exists() else None,
    trainable_backbone_layers=trainable_backbone_layers,
).to(device)

# Warm start from a previous detector checkpoint (e.g. VOC → COCO).
if warm_start_ckpt is not None:
    ws_path = PROJECT_ROOT / "checkpoints" / warm_start_ckpt
    if ws_path.exists():
        ws = torch.load(str(ws_path), map_location=device, weights_only=False)
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in ws["model_state_dict"].items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own_state)
        print(f"Warm-started from {warm_start_ckpt}: "
              f"loaded {loaded} keys, skipped {skipped} (shape/class mismatches)")
    else:
        print(f"(warning) warm-start ckpt not found: {ws_path}")

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_total:,}  Trainable: {n_train:,}")


# --- Optimizer -----------------------------------------------------------

def split_params(model):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.body.") else head_params).append(p)
    return backbone_params, head_params


bb_params, head_params = split_params(model)
print(f"Backbone trainable: {sum(p.numel() for p in bb_params):,}")
print(f"Head/FPN trainable: {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.AdamW([
    {"params": bb_params, "lr": lr_backbone},
    {"params": head_params, "lr": lr_head},
], weight_decay=weight_decay)


# --- LR schedule ---------------------------------------------------------

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * t))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# AMP setup
scaler = torch.amp.GradScaler("cuda") if use_amp else None


# --- Checkpointing -------------------------------------------------------

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / f"{EXPERIMENT_NAME}_best.pth"
start_epoch = 0
best_map = 0.0

if checkpoint_path.exists():
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_map = ckpt.get("map", 0.0)
    print(f"Resumed from epoch {ckpt['epoch']} (mAP: {best_map:.3f})")
else:
    print("No resume checkpoint — fresh start for this experiment.")


# --- Training ------------------------------------------------------------

def move_targets(targets, device):
    return [
        {
            "boxes": t["boxes"].to(device, non_blocking=True),
            "labels": t["labels"].to(device, non_blocking=True),
        }
        for t in targets
    ]


for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running = {"classification": 0.0, "bbox_regression": 0.0, "total": 0.0}
    n_batches = 0
    t0 = time.time()

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        if max_iters is not None and batch_idx >= max_iters:
            break

        imgs = [img.to(device, non_blocking=True) for img in imgs]
        targets = move_targets(targets, device)
        total_gt = sum(t["boxes"].shape[0] for t in targets)
        if total_gt == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
            optimizer.step()

        running["classification"] += loss_dict["classification"].item()
        running["bbox_regression"] += loss_dict["bbox_regression"].item()
        running["total"] += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            elapsed = time.time() - t0
            its = (batch_idx + 1) / max(1e-6, elapsed)
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0
            print(f"  iter {batch_idx}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  (cls {loss_dict['classification'].item():.4f} "
                  f"bbox {loss_dict['bbox_regression'].item():.4f})  "
                  f"{its:.2f} it/s  gpu_mem={mem_mb:.0f} MB")

    scheduler.step()

    if n_batches == 0:
        print("No batches processed — aborting epoch.")
        continue

    avg_cls = running["classification"] / n_batches
    avg_bbox = running["bbox_regression"] / n_batches
    avg_total = running["total"] / n_batches
    lr_bb = optimizer.param_groups[0]["lr"]
    lr_hd = optimizer.param_groups[1]["lr"]

    print(f"Epoch {epoch} train | total={avg_total:.4f} cls={avg_cls:.4f} bbox={avg_bbox:.4f} "
          f"| LR bb={lr_bb:.2e} head={lr_hd:.2e} | {n_batches} batches "
          f"in {time.time()-t0:.1f}s")

    # --- Eval ------------------------------------------------------------
    if do_eval:
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device, non_blocking=True) for img in imgs]
                targets = move_targets(targets, device)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        preds = model(imgs)
                else:
                    preds = model(imgs)
                preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
                targs_cpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
                metric.update(preds_cpu, targs_cpu)
        result = metric.compute()
        current_map = float(result["map_50"])
        map_all = float(result["map"])
        print(f"Epoch {epoch} eval | mAP@50={current_map:.3f}  mAP@[.5:.95]={map_all:.3f}")

        if current_map > best_map:
            best_map = current_map
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "map": best_map,
                "map_all": map_all,
                "classes": class_names,
                "experiment_name": EXPERIMENT_NAME,
                "num_steps": num_steps,
                "img_size": img_size,
                "dataset": dataset_name,
            }, checkpoint_path)
            print(f">> Best saved at epoch {epoch} (mAP@50: {best_map:.3f})")

print("Training complete.")
