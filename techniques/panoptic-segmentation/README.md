# Panoptic Segmentation (Reference §47.303)

Kirillov, He, Girshick, Rother & Dollár (2019). Unify semantic
("stuff" — road, sky) and instance ("things" — car, person)
segmentation into a single per-pixel `(category, instance_id)`
label. Panoptic Quality:

```
PQ = (Σ_TP IoU(p, g)) / (TP + 0.5 FP + 0.5 FN) = SQ · RQ
```

where SQ = segmentation quality (mean IoU on TPs) and RQ =
recognition quality (F1-like matching score).

## Files

- `python/panoptic_segmentation.py` — Compute panoptic quality
  on a 20×20 toy panoptic map. 3 classes (background, car,
  person); prediction is near-perfect on car #1 and person,
  missed car #2 (FN), and has a spurious extra car (FP). Mean
  PQ = 0.67 with per-class breakdown (SQ × RQ).
- `r/panoptic_segmentation.R` — reticulate + detectron2 /
  mmsegmentation (R); `detectron2` PanopticFPN / Mask2Former,
  `mmsegmentation` MaskFormer, `panopticapi` COCO PQ,
  from-scratch (Python).

## When to use

- **Autonomous driving** — road / lane (stuff) + car / person
  (things) in one output.
- **Urban scene understanding**, satellite imagery combining
  land cover + individual buildings.
- **When both semantic and instance segmentation are needed**
  without running two networks.

## When NOT to use

- **Pure instance segmentation** — Mask R-CNN suffices.
- **Pure semantic segmentation** — DeepLab / U-Net simpler.
- **Very small objects** — panoptic architectures need
  careful attention to size.

## Assumptions & caveats

- **PQ evaluation with IoU > 0.5 threshold** for TP; matches
  above threshold count.
- **"Stuff" categories share one instance ID** per category.
- **Modern architectures**: Mask2Former, MaskDINO handle
  panoptic elegantly with unified queries.
- **Void / unlabelled** — often ignored in PQ; check dataset
  convention.

## Related in this repo

- `u-net-segmentation`, `semantic-segmentation-fcn` —
  semantic-only precursors.
- `faster-rcnn-region-proposal` — Mask R-CNN's basis.
- `detr-transformer-detection` — Mask2Former / MaskDINO
  ancestor.
- `iou-generalized-iou` — the underlying overlap metric.

## Run

```
python techniques/panoptic-segmentation/python/panoptic_segmentation.py
Rscript techniques/panoptic-segmentation/r/panoptic_segmentation.R
```

**Refs:** Kirillov, A., He, K., Girshick, R., Rother, C. and Dollár, P. "Panoptic segmentation." In *CVPR*, 2019.

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
