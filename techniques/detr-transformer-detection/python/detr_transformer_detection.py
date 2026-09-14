"""DETR - Detection Transformer (Reference Sec 47.301).

Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko 2020 ECCV.
End-to-end object detection with a Transformer:

    CNN backbone -> flatten features -> Transformer encoder-decoder
    -> N=100 output "object queries" -> per-query (class, box)
    -> Hungarian matching between predictions and ground truth
    -> combined classification + L1 + GIoU box loss.

Eliminates hand-designed components (anchors, NMS) present in
Faster R-CNN / YOLO. Key insight: SET PREDICTION with permutation-
invariant loss via bipartite matching.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hungarian_match(cost_matrix):
    """Simple O(n^3) Hungarian via scipy.optimize.linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    return list(zip(row_idx, col_idx))


def giou_1d_approx(a, b):
    """1-D IoU proxy for a toy demo (interval overlap)."""
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / max(union, 1e-9)


if __name__ == "__main__":
    print("=== DETR - Detection Transformer (Carion et al 2020 ECCV) ===\n")

    # Toy: 3 ground truths, 5 predictions (2 "no object")
    gts = [{"cls": 1, "box": (0.1, 0.2)},
           {"cls": 2, "box": (0.4, 0.7)},
           {"cls": 1, "box": (0.75, 0.9)}]
    preds = [{"cls_probs": np.array([0.05, 0.90, 0.05]), "box": (0.11, 0.19)},
             {"cls_probs": np.array([0.10, 0.10, 0.80]), "box": (0.42, 0.65)},
             {"cls_probs": np.array([0.05, 0.85, 0.10]), "box": (0.72, 0.92)},
             {"cls_probs": np.array([0.70, 0.15, 0.15]), "box": (0.30, 0.35)},
             {"cls_probs": np.array([0.85, 0.10, 0.05]), "box": (0.50, 0.55)}]

    # Bipartite matching: cost = -prob(class) - IoU(box)
    C = np.zeros((len(preds), len(gts) + 2))                      # pad with 2 "no-obj"
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            C[i, j] = -p["cls_probs"][g["cls"]] - giou_1d_approx(p["box"], g["box"])
        # No-object cost = -prob(no-object class 0)
        for j in range(len(gts), len(gts) + 2):
            C[i, j] = -p["cls_probs"][0]

    print(f"  {len(preds)} predictions, {len(gts)} ground truths, "
          f"{len(gts) + 2} slots (padded no-obj)")
    matches = hungarian_match(C)
    print(f"\n  Hungarian matches (prediction -> ground truth):")
    for pi, gi in matches:
        if gi < len(gts):
            print(f"    pred {pi} -> gt {gi}  class {gts[gi]['cls']}   "
                  f"IoU = {giou_1d_approx(preds[pi]['box'], gts[gi]['box']):.3f}")
        else:
            print(f"    pred {pi} -> NO-OBJECT   prob(no-obj) = {preds[pi]['cls_probs'][0]:.3f}")

    print(f"\n  Set-prediction loss (per matched pair):")
    print(f"    L = -log p(cls_matched) + L1(box) + GIoU(box)")
    print(f"    'no-object' predictions incur only classification loss.")

    print(f"\n  No NMS needed: distinct queries produce distinct predictions.")

    print("\n--- library cross-check (transformers.DETR; facebookresearch/detr; mmdetection) ---")
