"""Training loop for SEW-RetinaNet on Pascal VOC or COCO.

Path A progression (V3 → V6):
  V3_imagenet : swap backbone to ImageNet-pretrained SEW-ResNet-18 weights.
  V4_mosaic   : V3 + Mosaic/MixUp strong augmentation.
  V5_kd       : V4 + KD from frozen ANN teacher (torchvision RetinaNet-R50).
  V6_spiking  : V5 + spike-native I-LIF FPN/head.

Single-file configuration at top, Tee logger, best-mAP checkpointing,
AdamW with warmup + cosine schedule, optional AMP.
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

from cifar100_spikedetect.model import build_retinanet


# --- Experiment config ---------------------------------------------------

EXPERIMENT = "V3_imagenet"

# schema: dataset, epochs, warmup, max_iters, do_eval,
#         backbone_source, warm_start_ckpt, warm_start_scope, aug_level, label
EXPERIMENTS = {
    "V3_debug": ("coco", 1, 0, 50, False,
                 'imagenet_ann', None, 'none', 'basic',
                 "V3 smoke test — 50 iters, no eval"),
    "V3_imagenet": ("coco", 20, 2, None, True,
                    'imagenet_ann',
                    "cifar100_detect_V2_coco_best.pth", 'head_fpn',
                    'basic',
                    "V3 ImageNet backbone + V2 head/FPN warm-start"),
    "V4_mosaic": ("coco", 20, 2, None, True,
                  'imagenet_ann',
                  "cifar100_spikedetect_V3_imagenet_best.pth", 'all',
                  'strong',
                  "V4 = V3 + Mosaic/MixUp"),
    "V5_kd": ("coco", 20, 2, None, True,
              'imagenet_ann',
              "cifar100_spikedetect_V4_mosaic_best.pth", 'all',
              'strong',
              "V5 = V4 + KD from ANN teacher (enable via USE_KD)"),
    "V6_spiking": ("coco", 15, 2, None, True,
                   'imagenet_ann',
                   "cifar100_spikedetect_V5_kd_best.pth", 'all',
                   'strong',
                   "V6 = V5 + spike-native FPN/head (I-LIF K=4)"),
}

(dataset_name, num_epochs, warmup_epochs, max_iters, do_eval,
 backbone_source, warm_start_ckpt, warm_start_scope, aug_level,
 exp_desc) = EXPERIMENTS[EXPERIMENT]
EXPERIMENT_NAME = f"cifar100_spikedetect_{EXPERIMENT}"

# --- Feature flags (enable stage-specific code paths) ------------------
USE_KD = EXPERIMENT in ("V5_kd",)
USE_SPIKING_HEAD = EXPERIMENT in ("V6_spiking",)


# --- Hyperparameters -----------------------------------------------------

SEED = 42
img_size = 416
num_steps = 8

if dataset_name == "coco":
    batch_size = 20
    num_workers = 10
    lr_head = 3e-4
    lr_backbone = 3e-5
    use_amp = True
else:
    batch_size = 8
    num_workers = 4
    lr_head = 5e-4
    lr_backbone = 5e-5
    use_amp = False

# For V6 (spiking head) use lower LR since we're fine-tuning warmed-up convs
if USE_SPIKING_HEAD:
    lr_head *= 0.33

weight_decay = 1e-4
grad_clip_norm = 1.0
trainable_backbone_layers = 3

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


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
print(f"dataset={dataset_name} batch={batch_size} img={img_size} "
      f"epochs={num_epochs} warmup={warmup_epochs} amp={use_amp}")
print(f"backbone_source={backbone_source}  warm_start={warm_start_ckpt} "
      f"scope={warm_start_scope}  aug={aug_level}")
print(f"USE_KD={USE_KD}  USE_SPIKING_HEAD={USE_SPIKING_HEAD}")


# --- Data ----------------------------------------------------------------

if dataset_name == "coco":
    from cifar100_spikedetect.data_coco import build_dataloaders as build_coco_loaders
    train_loader, val_loader, class_names = build_coco_loaders(
        root=str(PROJECT_ROOT / "data" / "coco"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        aug_level=aug_level,
    )
    num_classes = len(class_names)
else:
    from cifar100_spikedetect.data import build_dataloaders as build_voc_loaders, VOC_CLASSES
    train_loader, val_loader = build_voc_loaders(
        root=str(PROJECT_ROOT / "data" / "voc"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        download=False,
    )
    class_names = VOC_CLASSES
    num_classes = len(class_names)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
print(f"num_classes={num_classes} (incl. background)")


# --- Model ---------------------------------------------------------------

base_cifar_ckpt = PROJECT_ROOT / "checkpoints" / "cifar100_sew_T_best.pth"
model = build_retinanet(
    num_classes=num_classes,
    num_steps=num_steps,
    backbone_ckpt=str(base_cifar_ckpt) if base_cifar_ckpt.exists() else None,
    backbone_source=backbone_source,
    trainable_backbone_layers=trainable_backbone_layers,
    spiking_fpn=USE_SPIKING_HEAD,
    K=4,
).to(device)


# --- Warm-start from previous experiment checkpoint ---------------------

def warm_start_from(ckpt_path, scope='all'):
    """scope:
         'all' — load every matching tensor
         'head_fpn' — load only non-backbone tensors (FPN + classification/regression heads)
    """
    ws = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    src_state = ws["model_state_dict"]
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in src_state.items():
        if k not in own_state or own_state[k].shape != v.shape:
            skipped += 1
            continue
        if scope == 'head_fpn' and k.startswith('backbone.body.'):
            continue   # preserve the freshly-loaded ImageNet backbone
        own_state[k] = v
        loaded += 1
    model.load_state_dict(own_state)
    print(f"Warm-started from {ckpt_path.name} (scope={scope}): "
          f"loaded {loaded} keys, skipped {skipped}")


if warm_start_ckpt is not None:
    ws_path = PROJECT_ROOT / "checkpoints" / warm_start_ckpt
    if ws_path.exists():
        warm_start_from(ws_path, scope=warm_start_scope)
    else:
        print(f"(warning) warm-start ckpt not found: {ws_path}")

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_total:,}  Trainable: {n_train:,}")


# --- KD teacher (V5+) ---------------------------------------------------

teacher = None
KD_WEIGHT = 1.0  # coefficient on KD loss
if USE_KD:
    from cifar100_spikedetect.distill import (
        load_teacher, teacher_fpn_features,
        get_student_fpn_features, feature_kd_loss,
    )
    print("Loading ANN teacher: torchvision RetinaNet-R50-FPN-v2 @ 416...")
    teacher = load_teacher(device, min_size=img_size, max_size=img_size)


# --- Optimizer ----------------------------------------------------------

def split_params(model):
    bb, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (bb if name.startswith("backbone.body.") else head).append(p)
    return bb, head


bb_params, head_params = split_params(model)
print(f"Backbone trainable: {sum(p.numel() for p in bb_params):,}")
print(f"Head/FPN trainable: {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.AdamW([
    {"params": bb_params, "lr": lr_backbone},
    {"params": head_params, "lr": lr_head},
], weight_decay=weight_decay)


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * t))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda") if use_amp else None


# --- Checkpointing ------------------------------------------------------

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


# --- Training -----------------------------------------------------------

def move_targets(targets, device):
    return [{"boxes": t["boxes"].to(device, non_blocking=True),
             "labels": t["labels"].to(device, non_blocking=True)} for t in targets]


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

        # V4+: apply MixUp batch-wise with 50% probability
        if aug_level == "strong":
            from cifar100_spikedetect.augs import mixup_batch
            imgs, targets = mixup_batch(imgs, targets, prob=0.5, alpha=8.0)

        if sum(t["boxes"].shape[0] for t in targets) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                if USE_KD and teacher is not None:
                    t_feats = teacher_fpn_features(teacher, imgs)
                    s_feats = get_student_fpn_features(model, imgs)
                    kd_loss = feature_kd_loss(s_feats, t_feats)
                    loss = loss + KD_WEIGHT * kd_loss
                    loss_dict["kd"] = kd_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if USE_KD and teacher is not None:
                t_feats = teacher_fpn_features(teacher, imgs)
                s_feats = get_student_fpn_features(model, imgs)
                kd_loss = feature_kd_loss(s_feats, t_feats)
                loss = loss + KD_WEIGHT * kd_loss
                loss_dict["kd"] = kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
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
        print("No batches processed — aborting epoch."); continue

    avg_cls = running["classification"] / n_batches
    avg_bbox = running["bbox_regression"] / n_batches
    avg_total = running["total"] / n_batches
    lr_bb, lr_hd = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
    print(f"Epoch {epoch} train | total={avg_total:.4f} cls={avg_cls:.4f} bbox={avg_bbox:.4f} "
          f"| LR bb={lr_bb:.2e} head={lr_hd:.2e} | {n_batches} batches in {time.time()-t0:.1f}s")

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
                "map": best_map, "map_all": map_all,
                "classes": class_names,
                "experiment_name": EXPERIMENT_NAME,
                "num_steps": num_steps, "img_size": img_size,
                "dataset": dataset_name,
            }, checkpoint_path)
            print(f">> Best saved at epoch {epoch} (mAP@50: {best_map:.3f})")

print("Training complete.")
