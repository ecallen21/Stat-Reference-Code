"""FCOS - Fully Convolutional One-Stage Detection (Ref Sec 47.302).

Tian, Shen, Chen & He 2019 ICCV. Anchor-FREE detection: for
each feature-map location `p`, predict:

    - class probs
    - (l, t, r, b) offsets to the object's bounding box (if any)
    - centerness = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))

Locations OUTSIDE the object are labelled negative; multi-scale
FPN heads route large / small objects to appropriate feature
levels. Simpler than Faster R-CNN / RetinaNet, competitive AP.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def compute_ltrb(px, py, box):
    """Distances from point p to sides of box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    l = px - x1; t = py - y1
    r = x2 - px; b = y2 - py
    return l, t, r, b


def centerness(l, t, r, b):
    """FCOS centerness in [0, 1]; 1 at box centre, 0 at boundary."""
    lr = min(l, r) / max(max(l, r), 1e-9)
    tb = min(t, b) / max(max(t, b), 1e-9)
    return float(np.sqrt(max(lr * tb, 0)))


if __name__ == "__main__":
    print("=== FCOS Anchor-Free Detection (Tian et al 2019 ICCV) ===\n")

    # Ground truth box in image coords
    box = (50, 40, 150, 200)
    print(f"  Ground-truth box (x1, y1, x2, y2) = {box}\n")

    # Sample a grid of feature-map locations and report (l, t, r, b) + centerness
    print(f"  {'point':>10}  {'inside?':>7}   {'(l,t,r,b)':>25}   {'centerness':>10}")
    for px, py in [(60, 60), (100, 120), (130, 180), (170, 100), (95, 55), (99, 121)]:
        inside = box[0] <= px <= box[2] and box[1] <= py <= box[3]
        if inside:
            l, t, r, b = compute_ltrb(px, py, box)
            c = centerness(l, t, r, b)
            print(f"  ({px:>3},{py:>3})   {'yes':>7}   ({l:>4}, {t:>4}, {r:>4}, {b:>4})   {c:>10.3f}")
        else:
            print(f"  ({px:>3},{py:>3})   {'no':>7}   {'--':>25}   {'0.0 (bg)':>10}")

    print(f"\n  Centerness peaks at the true CENTRE (100, 120) and DECAYS toward")
    print(f"  the boundaries - used to down-weight low-quality predictions during")
    print(f"  inference (multiplied with class score before NMS).\n")

    # Multi-scale FPN routing: assign objects to feature levels by max side
    print(f"  FPN routing example:")
    print(f"    max(box side) = {max(box[2] - box[0], box[3] - box[1])} pixels")
    print(f"    -> assigned to feature level with matching stride (e.g. P4/stride 16)")
    print(f"    Objects at each level are those with max side in a pre-defined range.")

    print("\n--- library cross-check (mmdetection FCOS; adelaidet FCOS; detectron2 FCOS) ---")
