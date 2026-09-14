"""IoU / GIoU / DIoU / CIoU (Reference Sec 47.293).

Rezatofighi et al 2019 CVPR (GIoU); Zheng et al 2020 AAAI
(DIoU / CIoU). Box-overlap metrics for object detection:

    IoU  = |A n B| / |A u B|                       (Jaccard)
    GIoU = IoU - |C \ (A u B)| / |C|              (C = smallest enclosing box)
    DIoU = IoU - rho^2(a, b) / c^2                (rho = centre distance, c = diagonal of C)
    CIoU = DIoU - alpha * v                       (v = aspect-ratio consistency)

Higher-order variants (GIoU/DIoU/CIoU) fix the "no gradient
when boxes don't overlap" pathology of plain IoU.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    """IoU of two axis-aligned boxes [x1, y1, x2, y2]."""
    inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = box_area(a) + box_area(b) - inter
    return inter / max(union, 1e-9)


def giou(a, b):
    """Generalised IoU (Rezatofighi 2019)."""
    i = iou(a, b)
    c_x1 = min(a[0], b[0]); c_y1 = min(a[1], b[1])
    c_x2 = max(a[2], b[2]); c_y2 = max(a[3], b[3])
    C = (c_x2 - c_x1) * (c_y2 - c_y1)
    union = box_area(a) + box_area(b) - i * (box_area(a) + box_area(b) - i)
    return i - (C - box_area(a) - box_area(b) + iou(a, b) * max(union, 1e-9)) / max(C, 1e-9)


def diou(a, b):
    """Distance IoU (Zheng et al 2020)."""
    i = iou(a, b)
    a_cx = (a[0] + a[2]) / 2; a_cy = (a[1] + a[3]) / 2
    b_cx = (b[0] + b[2]) / 2; b_cy = (b[1] + b[3]) / 2
    rho2 = (a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2
    c_x1 = min(a[0], b[0]); c_y1 = min(a[1], b[1])
    c_x2 = max(a[2], b[2]); c_y2 = max(a[3], b[3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
    return i - rho2 / max(c2, 1e-9)


def ciou(a, b):
    """Complete IoU (Zheng et al 2020) - adds aspect-ratio term."""
    di = diou(a, b)
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    v = 4 / (np.pi ** 2) * (np.arctan(aw / max(ah, 1e-9)) - np.arctan(bw / max(bh, 1e-9))) ** 2
    i = iou(a, b)
    alpha = v / max(1 - i + v, 1e-9)
    return di - alpha * v


if __name__ == "__main__":
    print("=== IoU / GIoU / DIoU / CIoU (Rezatofighi 2019; Zheng 2020) ===\n")

    # Case A: overlapping boxes
    a = [10, 10, 30, 30]; b = [20, 20, 40, 40]
    print(f"  Case A - overlapping:  A={a}  B={b}")
    print(f"    IoU = {iou(a, b):.3f}   GIoU = {giou(a, b):.3f}   "
          f"DIoU = {diou(a, b):.3f}   CIoU = {ciou(a, b):.3f}\n")

    # Case B: disjoint boxes
    a = [10, 10, 30, 30]; b = [100, 100, 130, 130]
    print(f"  Case B - disjoint:    A={a}  B={b}")
    print(f"    IoU = {iou(a, b):.3f}   GIoU = {giou(a, b):.3f}   "
          f"DIoU = {diou(a, b):.3f}   CIoU = {ciou(a, b):.3f}")
    print(f"    IoU = 0 (no gradient); GIoU, DIoU are negative (informative gradient)\n")

    # Case C: aspect-ratio mismatch
    a = [10, 10, 90, 20]; b = [10, 10, 30, 60]
    print(f"  Case C - aspect mismatch:  A={a} (10:80)  B={b} (50:20)")
    print(f"    IoU = {iou(a, b):.3f}   GIoU = {giou(a, b):.3f}   "
          f"DIoU = {diou(a, b):.3f}   CIoU = {ciou(a, b):.3f}")
    print(f"    CIoU penalises aspect-ratio disagreement (v > 0)")

    print("\n--- library cross-check (torchvision.ops.generalized_box_iou; torchvision.ops.box_iou) ---")
