"""YOLO - You Only Look Once (Reference Sec 47.295).

Redmon, Divvala, Girshick & Farhadi 2016 CVPR. Single-stage
detector: divide the image into an S x S grid; each cell
predicts B bounding boxes (x, y, w, h, obj) plus C class
probabilities. Loss:

    L = lambda_coord * sum (x, y, w, h) errors        (only for boxes with objects)
      + sum obj_score errors                          (all boxes)
      + lambda_noobj * sum obj_score errors (no obj)
      + sum class errors

Fast enough for real-time. Later versions (v3-v8) add anchor
boxes, feature pyramids, and better backbones.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def yolo_loss(pred, target, lambda_coord=5.0, lambda_noobj=0.5):
    """Simplified YOLO loss on a single grid cell. pred/target:
    [x, y, w, h, obj, c0, c1, ...]. Sums squared errors with the
    lambda-weighted regime."""
    obj_mask = target[4] > 0.5
    loss = 0.0
    if obj_mask:
        loss += lambda_coord * ((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)
        loss += lambda_coord * ((np.sqrt(max(pred[2], 1e-6)) - np.sqrt(target[2])) ** 2 +
                                 (np.sqrt(max(pred[3], 1e-6)) - np.sqrt(target[3])) ** 2)
        loss += (pred[4] - target[4]) ** 2                        # obj confidence
        loss += float(np.sum((pred[5:] - target[5:]) ** 2))       # class error
    else:
        loss += lambda_noobj * (pred[4] - target[4]) ** 2
    return float(loss)


if __name__ == "__main__":
    print("=== YOLO (Redmon et al 2016 CVPR) ===\n")
    rng = np.random.default_rng(0)

    # Toy: 2-class detection over a 3x3 grid, B=1 box per cell.
    S, B, C = 3, 1, 2
    # Ground truth: object of class 0 at cell (1, 1), object of class 1 at cell (2, 0)
    target = np.zeros((S, S, 5 + C))
    target[1, 1] = [0.5, 0.5, 0.3, 0.4, 1.0, 1.0, 0.0]           # x,y within cell; w,h; obj; class[0]
    target[2, 0] = [0.6, 0.6, 0.5, 0.5, 1.0, 0.0, 1.0]

    # Prediction: near-perfect at (1, 1); wrong at (2, 0); false confidence at (0, 0)
    pred = np.zeros_like(target)
    pred[1, 1] = [0.52, 0.48, 0.28, 0.41, 0.9, 0.85, 0.15]
    pred[2, 0] = [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.7]
    pred[0, 0, 4] = 0.6                                          # false positive obj score

    print(f"  Grid {S}x{S}, {B} box per cell, {C} classes\n")
    total = 0.0
    for i in range(S):
        for j in range(S):
            l = yolo_loss(pred[i, j], target[i, j])
            has_obj = target[i, j, 4] > 0.5
            marker = "  <-- HAS OBJ" if has_obj else ""
            if l > 0.01 or has_obj:
                print(f"    cell ({i},{j})  loss = {l:.4f}{marker}")
            total += l
    print(f"\n  Total YOLO loss:  {total:.4f}")
    print(f"\n  lambda_coord = 5 emphasises coord errors; lambda_noobj = 0.5")
    print(f"  down-weights background-cell confidence errors (else the many empty cells dominate).")

    print("\n--- library cross-check (ultralytics YOLOv8; darknet original; mmdetection) ---")
