# Faster R-CNN Region Proposal Network (Reference §47.296)

Ren, He, Girshick & Sun (2015). Two-stage detector:

- **Stage 1 (RPN)**: sliding-window anchor prediction on a
  feature map — per anchor, output objectness + box offset.
  Keep top-N post-NMS proposals.
- **Stage 2 (Fast R-CNN)**: RoI pool / RoI align on each
  proposal, then classify + regress with a shared head.

RPN + detector share backbone features — the efficiency
improvement over R-CNN / Fast R-CNN.

## Files

- `python/faster_rcnn_region_proposal.py` — Anchor generation
  + RPN target assignment (positive if IoU ≥ 0.7, negative if
  < 0.3, ignored between). Demo: 14×14 feature map, 9 anchors
  per cell → 1764 anchors; two ground-truth boxes yield 4
  positive anchors and 1548 negatives; mini-batch typically
  samples 256 with 1:1 pos:neg.
- `r/faster_rcnn_region_proposal.R` — reticulate +
  `torchvision.models.detection.faster_rcnn` (R);
  `torchvision`, `mmdetection`, `detectron2` FasterRCNN,
  from-scratch (Python).

## When to use

- **High-accuracy detection** on standard benchmarks — Faster
  R-CNN + FPN + ResNet50 is a strong reproducible baseline.
- **Small object detection** — RoIAlign preserves detail
  better than YOLO's grid cells.
- **When a two-stage head is fine** — Mask R-CNN builds on
  the same recipe for segmentation.

## When NOT to use

- **Real-time deployment** — YOLO / EfficientDet is faster.
- **Extremely dense scenes** — DETR variants (DINO) beat two-
  stage detectors while being conceptually simpler.
- **Edge / mobile** — RPN + RoI pooling has high memory
  footprint.

## Assumptions & caveats

- **Anchor scales/ratios** — 3 scales × 3 aspect ratios is
  standard; tune per dataset.
- **RPN batch sampling** — 256 total with ~1:1 pos:neg;
  unbalanced batches wreck training.
- **Positive threshold 0.7** — lowered to 0.5 for very
  small objects.
- **RoI Align vs RoI Pool** — Align (He 2017 Mask R-CNN)
  preserves sub-pixel accuracy.

## Related in this repo

- `yolo-object-detection`, `anchor-free-fcos`,
  `detr-transformer-detection` — alternatives.
- `iou-generalized-iou` — box-overlap metric.
- `non-max-suppression-nms` — post-processing.
- `u-net-segmentation`, `panoptic-segmentation` — segmentation
  extensions of the same two-stage template.

## Run

```
python techniques/faster-rcnn-region-proposal/python/faster_rcnn_region_proposal.py
Rscript techniques/faster-rcnn-region-proposal/r/faster_rcnn_region_proposal.R
```

**Refs:** Ren, S., He, K., Girshick, R. and Sun, J. "Faster R-CNN: Towards real-time object detection with region proposal networks." In *NIPS*, pp. 91-99, 2015.

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
