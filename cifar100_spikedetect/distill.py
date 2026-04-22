"""Knowledge distillation from an ANN teacher detector.

Strategy: feature-level FPN distillation. Both student and teacher use
torchvision-style RetinaNet with the same FPN output channels (256) and
comparable FPN levels. We align feature maps at each level and minimize
MSE between them.

Why feature-level (not logit-level):
  - Torchvision's RetinaNet in eval mode runs NMS and score-filtering, so
    raw per-anchor logits are not directly exposed. Feature-level KD avoids
    that and directly shapes the student's representations.
  - Empirically, feature-level KD yields similar gains (+2-4 mAP) to logit
    KD on small detection models.

Usage:
    teacher = load_teacher(device)   # frozen RetinaNet-ResNet50-FPN-v2
    kd_loss = feature_kd_loss(student_feats, teacher_feats)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn_v2


def load_teacher(device, weights='DEFAULT', min_size=416, max_size=416):
    """Load a frozen pretrained ANN RetinaNet-ResNet50-FPN-v2 teacher.

    Override the internal GeneralizedRCNNTransform min/max size so the
    teacher's FPN feature maps match the student's spatial sizes (avoiding
    the lossy resize during KD loss).
    """
    teacher = retinanet_resnet50_fpn_v2(weights=weights).to(device)
    # Shrink teacher's internal image-resize step to match our 416 training.
    teacher.transform.min_size = (min_size,)
    teacher.transform.max_size = max_size
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def teacher_fpn_features(teacher, images):
    """Run teacher's backbone + FPN on images, return OrderedDict of feats.

    The teacher expects images as a list of float tensors in ~[0, 1] range.
    Its internal GeneralizedRCNNTransform normalizes with ImageNet mean/std.
    """
    # Pass through teacher.transform to match its expected format
    original_sizes = [img.shape[-2:] for img in images]
    images_list, _ = teacher.transform(images, None)
    # teacher.backbone is BackboneWithFPN → returns OrderedDict
    return teacher.backbone(images_list.tensors)


def feature_kd_loss(student_feats, teacher_feats, weight_per_level=None):
    """MSE between student and teacher FPN features at matching levels.

    If spatial sizes don't match, bilinearly resize the teacher to the
    student's size before comparing. Both dicts should have the same keys
    (torchvision FPN convention: '0','1','2','p6','p7').
    """
    loss = 0.0
    count = 0
    for key, s_feat in student_feats.items():
        if key not in teacher_feats:
            continue
        t_feat = teacher_feats[key]
        # Resize teacher to student spatial size if needed
        if t_feat.shape[-2:] != s_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:],
                                   mode='bilinear', align_corners=False)
        w = 1.0 if weight_per_level is None else weight_per_level.get(key, 1.0)
        loss = loss + w * F.mse_loss(s_feat, t_feat)
        count += 1
    return loss / max(count, 1)


def get_student_fpn_features(student_model, images):
    """Run student's backbone + FPN to get the OrderedDict of features.

    student_model is our torchvision-wrapped RetinaNet with SEWBackboneWithFPN.
    """
    # Apply same transform as the student does internally
    original_sizes = [img.shape[-2:] for img in images]
    images_list, _ = student_model.transform(images, None)
    return student_model.backbone(images_list.tensors)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(device)
    imgs = [torch.randn(3, 416, 416, device=device) for _ in range(2)]
    feats = teacher_fpn_features(teacher, imgs)
    for k, v in feats.items():
        print(f"  teacher {k}: {tuple(v.shape)}")
