# FCOS — Anchor-Free Detection (Reference §47.302)

Tian, Shen, Chen & He (2019). Anchor-FREE detection: for
each feature-map location `p`, predict:

- class probabilities
- (l, t, r, b) offsets to the object's bounding box (if any)
- centerness = √( min(l, r)/max(l, r) · min(t, b)/max(t, b) )

Locations outside the object are labelled negative; multi-
scale FPN heads route large / small objects to appropriate
feature levels. Simpler than Faster R-CNN / RetinaNet and
competitive AP.

## Files

- `python/anchor_free_fcos.py` — Per-point (l, t, r, b)
  regression + centerness formula on a single toy box.
  Interior points get positive (l, t, r, b) and centerness
  ∈ (0, 1] peaking at the box centre; exterior points are
  background. FPN routing example based on max side.
- `r/anchor_free_fcos.R` — reticulate + mmdetection FCOS
  (R); `mmdetection`, `detectron2.projects.FCOS`,
  `adelaidet`, from-scratch (Python).

## When to use

- **Anchor-free detection** without the need to tune
  anchor scales / aspect ratios.
- **Dense point prediction** naturally extends to keypoint
  detection.
- **Multi-scale FPN routing** — different object sizes handled
  by dedicated feature levels.

## When NOT to use

- **Very small objects** — centerness weighting can hurt if
  box is only a few pixels wide.
- **Extremely fast inference** — YOLO variants can be faster.
- **Where DETR shines** — dense scenes, occlusion, complex
  layouts.

## Assumptions & caveats

- **FPN levels + range assignment** — each level handles
  objects with `max(side)` in a pre-defined range.
- **Centerness clip** — during inference, multiply
  class score × centerness before NMS.
- **Positive assignment ambiguity** — the centerness weighting
  gracefully handles overlapping objects.
- **Loss** — Focal (classification) + IoU (regression) +
  BCE (centerness).

## Related in this repo

- `yolo-object-detection`, `faster-rcnn-region-proposal`,
  `detr-transformer-detection` — alternatives.
- `iou-generalized-iou` — regression loss.
- `focal-loss-imbalance` — classification loss.
- `non-max-suppression-nms` — post-processing.

## Run

```
python techniques/anchor-free-fcos/python/anchor_free_fcos.py
Rscript techniques/anchor-free-fcos/r/anchor_free_fcos.R
```

**Refs:** Tian, Z., Shen, C., Chen, H. and He, T. "FCOS: Fully convolutional one-stage object detection." In *ICCV*, 2019.

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
