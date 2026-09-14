# YOLO Object Detection (Reference §47.295)

Redmon, Divvala, Girshick & Farhadi (2016). Single-stage
detector: divide image into `S × S` grid; each cell predicts
`B` boxes (x, y, w, h, obj) + `C` class probabilities. Loss:

```
L = λ_coord · Σ (x, y, w, h) errors    (only object cells)
  + Σ obj errors                        (all boxes)
  + λ_noobj · Σ obj errors (no-obj)
  + Σ class errors
```

Fast enough for real-time. Later versions (v3-v8) add anchor
boxes, feature pyramids, and better backbones.

## Files

- `python/yolo_object_detection.py` — Simplified YOLO loss
  on a 3×3 grid, 2 classes, B=1. Demo shows loss contributions
  from cells with objects (weighted by λ_coord=5) vs cells
  with false-positive confidence (weighted by λ_noobj=0.5).
- `r/yolo_object_detection.R` — reticulate + `ultralytics`
  (R); `ultralytics` (YOLOv8/v11), `darknet` (Redmon),
  `mmdetection` YOLOv3, from-scratch (Python).

## When to use

- **Real-time detection** — YOLO ranks first for
  frames-per-second among modern detectors.
- **Deployment on edge / mobile** — smaller variants (YOLO-Nano,
  YOLOv8n) run on CPU / mobile GPU.
- **Multi-class detection with a modest number of classes**.

## When NOT to use

- **Very small objects** — anchor-based single-shot detectors
  struggle; feature pyramid + anchor-free (FCOS) or DETR
  may do better.
- **Highest-possible mAP** on well-benchmarked datasets —
  DINO / Mask2Former / two-stage variants can outperform.
- **When you need instance segmentation masks** — use YOLOv8-seg
  or Mask R-CNN.

## Assumptions & caveats

- **Anchor boxes** (v2+) — carefully tuned to the dataset via
  k-means on training boxes.
- **Grid ambiguity** — small objects share a cell; multi-scale
  feature pyramid (v3+) helps.
- **Confidence calibration** — YOLO scores are not
  well-calibrated by default; combine with temperature
  scaling if downstream logic needs probabilities.
- **Data augmentation** — mosaic + RandAugment critical to
  reach paper accuracy.

## Related in this repo

- `faster-rcnn-region-proposal` — two-stage cousin.
- `anchor-free-fcos`, `detr-transformer-detection` — modern
  alternatives.
- `non-max-suppression-nms` — required post-processing.
- `iou-generalized-iou` — the box-regression metric.

## Run

```
python techniques/yolo-object-detection/python/yolo_object_detection.py
Rscript techniques/yolo-object-detection/r/yolo_object_detection.R
```

**Refs:** Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. "You only look once: Unified, real-time object detection." In *CVPR*, 2016.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
