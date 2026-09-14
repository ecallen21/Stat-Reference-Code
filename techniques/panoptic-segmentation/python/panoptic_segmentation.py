"""Panoptic Segmentation (Reference Sec 47.303).

Kirillov, He, Girshick, Rother & Dollar 2019 CVPR. Unify
semantic ("stuff" - road, sky) and instance ("things" - car,
person) segmentation into a single per-pixel label:

    label(p) = (category, instance_id)

Each pixel gets a unique (cat, id) pair; "stuff" classes share
one instance id per category. Evaluation via PANOPTIC QUALITY:

    PQ = (sum_TP IoU(p, g) / (TP + 0.5 FP + 0.5 FN))
       = SQ * RQ                                   (segmentation * recognition)
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def panoptic_quality(pred, target, n_classes, iou_thresh=0.5):
    """PQ over a pred / target panoptic map (H, W) of (cat, id) tuples.
    pred/target: arrays of shape (H, W, 2) with (cat, id) integers."""
    pq_per_class = []
    for c in range(n_classes):
        # Extract instance ids per class
        gt_ids = set()
        pred_ids = set()
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i, j, 0] == c: gt_ids.add(int(target[i, j, 1]))
                if pred[i, j, 0] == c: pred_ids.add(int(pred[i, j, 1]))
        gt_ids.discard(-1); pred_ids.discard(-1)

        tp = 0; fp = 0; fn = 0; iou_sum = 0.0
        matched_pred = set()
        for gi in gt_ids:
            best_iou = 0; best_pi = None
            gt_mask = (target[..., 0] == c) & (target[..., 1] == gi)
            for pi in pred_ids:
                if pi in matched_pred: continue
                pr_mask = (pred[..., 0] == c) & (pred[..., 1] == pi)
                inter = float(np.sum(gt_mask & pr_mask))
                union = float(np.sum(gt_mask | pr_mask))
                if union == 0: continue
                ioui = inter / union
                if ioui > best_iou:
                    best_iou = ioui; best_pi = pi
            if best_pi is not None and best_iou > iou_thresh:
                tp += 1; iou_sum += best_iou; matched_pred.add(best_pi)
            else:
                fn += 1
        fp = len(pred_ids) - len(matched_pred)
        if tp + fp + fn == 0: continue
        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        pq_per_class.append((sq, rq, sq * rq))
    return pq_per_class


if __name__ == "__main__":
    print("=== Panoptic Segmentation (Kirillov et al 2019 CVPR) ===\n")

    # Toy 20x20 with 3 classes: 0=background (stuff), 1=car (thing), 2=person (thing)
    H = 20; C = 3
    target = np.zeros((H, H, 2), dtype=int)                       # (class, instance_id)
    target[..., 0] = 0; target[..., 1] = 0                        # background id = 0

    # Ground truth: 2 cars, 1 person
    target[3:9, 3:9, 0] = 1; target[3:9, 3:9, 1] = 1              # car instance 1
    target[12:18, 12:18, 0] = 1; target[12:18, 12:18, 1] = 2      # car instance 2
    target[3:7, 12:17, 0] = 2; target[3:7, 12:17, 1] = 3          # person instance 3

    # Prediction: near-perfect on car 1, missed car 2 (FN), extra spurious car (FP)
    pred = target.copy()
    # Missing car 2
    pred[12:18, 12:18, 0] = 0; pred[12:18, 12:18, 1] = 0
    # Spurious car at (16, 3)
    pred[16:19, 3:6, 0] = 1; pred[16:19, 3:6, 1] = 4
    # Person slightly offset
    pred[3:7, 12:17, 0] = 0; pred[3:7, 12:17, 1] = 0
    pred[3:7, 13:18, 0] = 2; pred[3:7, 13:18, 1] = 3

    pq = panoptic_quality(pred, target, C)
    print(f"  {'class':<10}  {'SQ':>6}  {'RQ':>6}  {'PQ':>6}")
    class_names = ["background", "car", "person"]
    for c, (sq, rq, pqc) in enumerate(pq):
        print(f"  {class_names[c]:<10}  {sq:>6.3f}  {rq:>6.3f}  {pqc:>6.3f}")

    mean_pq = np.mean([p[2] for p in pq])
    print(f"\n  mean PQ = {mean_pq:.3f}")
    print(f"    (SQ = seg quality on TPs; RQ = recognition quality; PQ = product)")

    print("\n--- library cross-check (panopticapi (COCO); mmsegmentation; detectron2 PanopticFPN) ---")
