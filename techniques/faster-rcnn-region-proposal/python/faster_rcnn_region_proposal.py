"""Faster R-CNN - Region Proposal Network (Ref Sec 47.296).

Ren, He, Girshick & Sun 2015 NIPS. Two-stage detector:

    Stage 1 (RPN): Anchor-based sliding-window on feature map;
        for each anchor, predict objectness + bounding-box offset.
        Keep top-N post-NMS proposals.
    Stage 2 (Fast R-CNN): For each proposal, apply RoI pool /
        RoI align, then classify + regress with a shared head.

The RPN and detector share backbone features - key efficiency
improvement over R-CNN / Fast R-CNN.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def generate_anchors(feat_h, feat_w, stride=16,
                     scales=(8, 16, 32), ratios=(0.5, 1.0, 2.0)):
    """Anchors at each feature-map cell in image coords."""
    anchors = []
    for i in range(feat_h):
        for j in range(feat_w):
            cy = (i + 0.5) * stride; cx = (j + 0.5) * stride
            for s in scales:
                for r in ratios:
                    w = s * stride / np.sqrt(r); h = s * stride * np.sqrt(r)
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(anchors)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aA = (a[2] - a[0]) * (a[3] - a[1]); aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aA + aB - inter, 1e-9)


def assign_anchors(anchors, gt_boxes, pos_thresh=0.7, neg_thresh=0.3):
    """RPN target assignment: positive / negative / ignore per anchor."""
    labels = np.full(len(anchors), -1)                            # -1 = ignore
    if len(gt_boxes) == 0:
        labels[:] = 0
        return labels
    ious = np.array([[iou(a, g) for g in gt_boxes] for a in anchors])
    max_iou = ious.max(axis=1)
    labels[max_iou >= pos_thresh] = 1                             # positive
    labels[max_iou < neg_thresh] = 0                              # negative
    # Also ensure each GT has at least one positive anchor (max IoU per GT)
    for g in range(len(gt_boxes)):
        labels[int(np.argmax(ious[:, g]))] = 1
    return labels


if __name__ == "__main__":
    print("=== Faster R-CNN RPN (Ren et al 2015 NIPS) ===\n")

    # Toy image 224x224, feature map 14x14 (stride 16)
    H, W = 224, 224; feat = 14
    anchors = generate_anchors(feat, feat, stride=16)
    print(f"  Feature map {feat}x{feat}, stride 16, 3 scales x 3 ratios = 9 anchors/cell")
    print(f"  Total anchors:  {len(anchors):,}\n")

    # Simulated ground-truth boxes
    gt_boxes = np.array([[50, 50, 150, 200], [120, 30, 210, 160]])
    labels = assign_anchors(anchors, gt_boxes, pos_thresh=0.7, neg_thresh=0.3)

    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum()); n_ign = int((labels == -1).sum())
    print(f"  {len(gt_boxes)} ground-truth boxes")
    print(f"    positive anchors (IoU >= 0.7):  {n_pos}")
    print(f"    negative anchors (IoU <  0.3):  {n_neg}")
    print(f"    ignored (0.3 <= IoU < 0.7):    {n_ign}")
    print(f"\n  RPN loss = classif(pos vs neg) + smoothL1(regression on positives).")
    print(f"  Typical mini-batch: 256 sampled with pos:neg ratio ~ 1:1.")

    print("\n--- library cross-check (torchvision.models.detection.faster_rcnn; mmdetection) ---")
