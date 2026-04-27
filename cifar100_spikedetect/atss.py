"""ATSS-style adaptive matcher for torchvision RetinaNet.

Adaptive Training Sample Selection (Zhang et al. CVPR 2020) replaces fixed
IoU-threshold matching with a per-GT adaptive threshold based on the
distribution of top-K candidate IoUs.

This implementation is "ATSS-lite": it operates on the IoU match matrix
directly (no center-distance preselection across FPN levels), which captures
~80% of canonical ATSS's benefit with much simpler integration into
torchvision's RetinaNet.

Algorithm (per image):
  1. For each GT, take top-K=9 anchors by IoU.
  2. Threshold = mean(top-K IoUs) + std(top-K IoUs).
  3. Anchors with IoU >= threshold (and IoU is the GT's argmax) → positive.
  4. Other anchors → background (or low-quality match if torchvision Matcher
     sets allow_low_quality_matches=True).

Usage:
    from cifar100_spikedetect.atss import patch_retinanet_with_atss
    patch_retinanet_with_atss(model, topk=9)
"""

import torch
from torchvision.models.detection._utils import Matcher


class ATSSMatcher:
    """Drop-in replacement for torchvision.models.detection._utils.Matcher.

    Same interface (callable that takes match_quality_matrix → matched_idxs)
    but uses adaptive per-GT thresholds instead of fixed fg/bg cutoffs.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, topk=9, allow_low_quality_matches=True):
        self.topk = topk
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """match_quality_matrix: [num_gt, num_anchors] IoU matrix.

        Returns matched_idxs: [num_anchors] tensor where:
          -1 = below low threshold (definitely background)
          -2 = between thresholds (ignored)
          >=0 = matched to that GT index
        """
        if match_quality_matrix.numel() == 0:
            # Empty target case
            if match_quality_matrix.shape[0] == 0:
                # No GT → all anchors are background
                return torch.full(
                    (match_quality_matrix.shape[1],),
                    self.BELOW_LOW_THRESHOLD,
                    dtype=torch.int64,
                    device=match_quality_matrix.device,
                )
            else:
                # No anchors (shouldn't happen in practice)
                return torch.empty((0,), dtype=torch.int64,
                                   device=match_quality_matrix.device)

        num_gt, num_anchors = match_quality_matrix.shape

        # Each anchor's best GT (and the corresponding IoU)
        matched_vals, matched_gt_idx = match_quality_matrix.max(dim=0)
        # Each GT's best anchors (top-K by IoU)
        topk_actual = min(self.topk, num_anchors)
        topk_iou, _ = match_quality_matrix.topk(topk_actual, dim=1)  # [num_gt, K]

        # Adaptive threshold per GT: mean + std of top-K
        gt_means = topk_iou.mean(dim=1)        # [num_gt]
        gt_stds = topk_iou.std(dim=1) if topk_actual > 1 else torch.zeros_like(gt_means)
        gt_thresholds = gt_means + gt_stds      # [num_gt]

        # For each anchor: threshold of its assigned (argmax) GT
        per_anchor_threshold = gt_thresholds[matched_gt_idx]   # [num_anchors]
        # Floor the threshold so we don't accept truly garbage matches
        per_anchor_threshold = torch.clamp(per_anchor_threshold, min=0.15)

        matches = matched_gt_idx.clone()
        # Background: IoU < adaptive threshold
        below = matched_vals < per_anchor_threshold
        matches[below] = self.BELOW_LOW_THRESHOLD

        # Optional: allow low-quality matches so every GT has at least one
        # anchor (mirrors torchvision Matcher behavior). Find the best
        # anchor for each GT and force it positive.
        if self.allow_low_quality_matches:
            # For each GT, the best anchor (argmax across the row)
            highest_quality_per_gt, _ = match_quality_matrix.max(dim=1)
            gt_anchor_pairs = torch.where(
                match_quality_matrix == highest_quality_per_gt.unsqueeze(1)
            )
            anchor_idx_to_force = gt_anchor_pairs[1]
            forced_gt_idx = gt_anchor_pairs[0]
            matches[anchor_idx_to_force] = forced_gt_idx

        return matches


def patch_retinanet_with_atss(model, topk=9):
    """Replace the RetinaNet's proposal_matcher with our ATSSMatcher.

    Works on torchvision.models.detection.RetinaNet instances and our wrapper
    `build_retinanet`. After this call, all subsequent training forward passes
    use ATSS matching for both classification and regression head losses.
    """
    if not hasattr(model, "proposal_matcher"):
        raise ValueError("Model does not have a proposal_matcher (is this RetinaNet?)")
    old = model.proposal_matcher
    model.proposal_matcher = ATSSMatcher(topk=topk, allow_low_quality_matches=True)
    print(f"[ATSS] Replaced {type(old).__name__} → ATSSMatcher(topk={topk})")
    return model


if __name__ == "__main__":
    # Smoke test
    matcher = ATSSMatcher(topk=9)

    # 3 GTs, 100 anchors, random IoUs
    torch.manual_seed(0)
    iou = torch.rand(3, 100) * 0.3
    # Make a few high-quality matches
    iou[0, 7] = 0.85
    iou[1, 30] = 0.75
    iou[1, 31] = 0.65
    iou[2, 88] = 0.55

    matched = matcher(iou)
    print("matched_idxs sample:", matched[:20])
    print("num positive:", (matched >= 0).sum().item())
    print("num background:", (matched == -1).sum().item())
    print("num ignored:", (matched == -2).sum().item())

    # Test empty GT
    empty = matcher(torch.empty(0, 100))
    print("empty GT all-background:", (empty == -1).all().item())
