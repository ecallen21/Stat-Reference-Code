# Non-Maximum Suppression (Reference §47.294)

Neubeck & Van Gool (2006). Suppress overlapping bounding-box
detections keeping only the highest-scoring per cluster:

```
order boxes by score desc
for each remaining box b:
    keep b; remove all boxes with IoU(b, b') > threshold
```

**Soft-NMS** (Bodla et al 2017) decays scores instead of hard
deletion, improving recall in crowded scenes.

## Files

- `python/non_max_suppression_nms.py` — Hard NMS and Gaussian
  soft-NMS. Demo: 6 boxes clustered near 2 true objects. Hard
  NMS at IoU=0.5 keeps 3 boxes (the two peaks and the spurious
  low-conf); soft-NMS decays overlapping scores gracefully.
- `r/non_max_suppression_nms.R` — reticulate + torchvision.ops
  (R); torchvision.ops.nms / batched_nms, mmcv.ops.soft_nms,
  from-scratch (Python).

## When to use

- **Detector post-processing** — remove duplicate boxes.
- **Crowded scenes** — soft-NMS instead of hard NMS.
- **DETR-style set predictors** do NOT need NMS.

## When NOT to use

- **Anchor-free single-object detection** (segmentation
  heads with 1 output) — no need.
- **DETR / end-to-end set-prediction detectors** — Hungarian
  matching removes the need for NMS.
- **When ground-truth is a full mask** — use panoptic
  post-processing.

## Assumptions & caveats

- **IoU threshold** — 0.5 is COCO default; 0.7 for
  Faster-RCNN.
- **Class-wise vs class-agnostic NMS** — batched_nms with
  per-class treatment is safer.
- **Soft-NMS σ** — 0.5 is a common default for gaussian
  variant.
- **Score threshold** — apply a minimum-score filter before
  NMS to reduce work.

## Related in this repo

- `iou-generalized-iou` — the underlying overlap metric.
- `yolo-object-detection`, `faster-rcnn-region-proposal`,
  `anchor-free-fcos` — detectors that require NMS.
- `detr-transformer-detection` — NMS-free alternative.

## Run

```
python techniques/non-max-suppression-nms/python/non_max_suppression_nms.py
Rscript techniques/non-max-suppression-nms/r/non_max_suppression_nms.R
```

**Refs:** Neubeck, A. and Van Gool, L. "Efficient non-maximum suppression." In *ICPR*, 2006; Bodla, N., Singh, B., Chellappa, R. and Davis, L.S. "Soft-NMS — improving object detection with one line of code." In *ICCV*, 2017.

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
