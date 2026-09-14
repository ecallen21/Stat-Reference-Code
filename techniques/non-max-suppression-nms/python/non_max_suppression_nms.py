"""Non-Maximum Suppression - NMS (Reference Sec 47.294).

Neubeck & Van Gool 2006 ICPR. Suppress overlapping bounding-box
detections keeping only the highest-scoring per cluster:

    order boxes by score desc
    for each remaining box b:
        keep b; remove all boxes with IoU(b, b') > threshold

SOFT-NMS (Bodla 2017 ICCV) decays scores instead of hard
deletion, improving recall on crowded scenes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aA = (a[2] - a[0]) * (a[3] - a[1]); aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aA + aB - inter, 1e-9)


def nms(boxes, scores, thresh=0.5):
    """Return indices of KEPT boxes after hard NMS."""
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]; keep.append(i)
        rest = idx[1:]
        idx = np.array([j for j in rest if iou(boxes[i], boxes[j]) <= thresh])
    return keep


def soft_nms(boxes, scores, sigma=0.5, thresh=0.001, method="gaussian"):
    """Bodla et al 2017 soft-NMS. Returns kept indices + decayed scores."""
    n = len(boxes)
    boxes = list(boxes); scores = list(scores); order = list(range(n))
    kept = []
    while order:
        # Pick max
        max_i = int(np.argmax([scores[i] for i in order]))
        i = order.pop(max_i)
        kept.append((i, scores[i]))
        for k in order:
            iou_ik = iou(boxes[i], boxes[k])
            if method == "linear":
                if iou_ik > 0.3: scores[k] *= (1 - iou_ik)
            else:                                                # gaussian
                scores[k] *= np.exp(-iou_ik ** 2 / sigma)
        order = [k for k in order if scores[k] > thresh]
    return kept


if __name__ == "__main__":
    print("=== Non-Maximum Suppression (Neubeck-Van Gool 2006) ===\n")

    # 6 boxes around 2 true objects
    boxes = np.array([
        [10, 10, 30, 30], [12, 12, 32, 32], [11, 11, 31, 31],    # object 1
        [80, 80, 100, 100], [82, 82, 102, 102],                    # object 2
        [50, 50, 60, 60]                                            # spurious low-conf
    ])
    scores = np.array([0.9, 0.85, 0.75, 0.80, 0.70, 0.30])

    print(f"  Input: {len(boxes)} detections with scores {scores.tolist()}\n")

    print(f"  Hard NMS (IoU threshold = 0.5):")
    keep = nms(boxes, scores, thresh=0.5)
    for k in keep:
        print(f"    keep #{k}   box = {boxes[k].tolist()}   score = {scores[k]:.2f}")

    print(f"\n  Soft NMS (gaussian sigma = 0.5):")
    for i, s in soft_nms(boxes, scores.copy()):
        marker = "  <-- kept" if s > 0.3 else ""
        print(f"    box #{i}   score = {s:.3f}{marker}")

    print(f"\n  Hard NMS DROPS overlapping detections; soft NMS DECAYS them.")
    print(f"  Soft NMS is preferred when nearby objects are common (crowded scenes).")

    print("\n--- library cross-check (torchvision.ops.nms; mmcv.ops.soft_nms) ---")
