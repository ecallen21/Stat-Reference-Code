# IoU / GIoU / DIoU / CIoU (Reference §47.293)

Rezatofighi et al (2019, GIoU); Zheng et al (2020, DIoU / CIoU).
Box-overlap metrics for object detection:

```
IoU  = |A ∩ B| / |A ∪ B|                        (Jaccard)
GIoU = IoU − |C \ (A ∪ B)| / |C|                 (C = smallest enclosing box)
DIoU = IoU − ρ²(a, b) / c²                       (centre distance / diagonal)
CIoU = DIoU − α · v                              (aspect-ratio penalty)
```

Higher-order variants fix the "no gradient when boxes do not
overlap" pathology of plain IoU.

## Files

- `python/iou_generalized_iou.py` — closed-form IoU / GIoU /
  DIoU / CIoU on axis-aligned boxes. Cases: overlapping
  (IoU=0.14, GIoU=0.14), disjoint (IoU=0, GIoU=−0.91 provides
  gradient), aspect-mismatch (CIoU penalises via v).
- `r/iou_generalized_iou.R` — reticulate + torchvision.ops
  (R); torchvision.ops.box_iou / generalized_box_iou /
  distance_box_iou / complete_box_iou, from-scratch (Python).

## When to use

- **Object-detection training loss** — regress boxes with
  GIoU / DIoU / CIoU instead of L1 / L2.
- **Evaluating detector output** — mAP metrics use IoU.
- **Reporting** — always show IoU thresholds (0.5, 0.75, 0.5:0.95).

## When NOT to use

- **Rotated / oriented boxes** — need Rotated IoU or
  polygon-IoU.
- **3-D boxes** — 3-D IoU / KL-IoU are the analogues.
- **Very sparse detections** — L1 or Huber loss may be more
  stable when boxes are far from the target.

## Assumptions & caveats

- **Axis-aligned assumption** — the standard family; skewed
  boxes need shapely / cv2.
- **Numerical guards** — divisions by zero handled with `eps`.
- **CIoU α term** — non-differentiable at IoU=1; libraries
  freeze α when computing gradients.
- **Empirical guide** — CIoU is usually best for bounding-box
  regression; GIoU / DIoU are close.

## Related in this repo

- `non-max-suppression-nms` — post-processing that uses IoU.
- `yolo-object-detection`, `faster-rcnn-region-proposal`,
  `anchor-free-fcos`, `detr-transformer-detection` — all use
  IoU-family loss.
- `dice-loss-segmentation` — segmentation analogue.

## Run

```
python techniques/iou-generalized-iou/python/iou_generalized_iou.py
Rscript techniques/iou-generalized-iou/r/iou_generalized_iou.R
```

**Refs:** Rezatofighi, H. et al. "Generalized intersection over union: A metric and a loss for bounding box regression." In *CVPR*, 2019; Zheng, Z. et al. "Distance-IoU loss: Faster and better learning for bounding box regression." In *AAAI*, 2020.

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
