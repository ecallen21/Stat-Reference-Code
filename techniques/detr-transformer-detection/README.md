# DETR — Detection Transformer (Reference §47.301)

Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko (2020).
End-to-end object detection with a Transformer:

```
CNN backbone → flatten → encoder–decoder → N object queries
→ per-query (class, box) → Hungarian match to ground truth
→ cls + L1 + GIoU box loss.
```

Eliminates hand-designed components (anchors, NMS). Key
insight: SET PREDICTION with permutation-invariant loss via
bipartite matching.

## Files

- `python/detr_transformer_detection.py` — Hungarian matching
  via `scipy.optimize.linear_sum_assignment`. Demo: 3 ground
  truths, 5 predictions (2 no-object). Bipartite match pairs
  each prediction with either a ground truth or "no-object";
  IoU + class-prob cost surfaces the correct assignments.
- `r/detr_transformer_detection.R` — reticulate +
  `transformers.DetrForObjectDetection` (R); `transformers`,
  `facebookresearch/detr`, `mmdetection` DETR / Deformable
  DETR / DINO, from-scratch (Python).

## When to use

- **End-to-end training** — no NMS or anchor tuning.
- **Global reasoning** — attention over all image tokens
  helps with occlusion / interaction.
- **Foundation for extensions** — Deformable DETR, DINO,
  Mask DINO for segmentation.

## When NOT to use

- **Small-object detection with early DETR** — slow
  convergence; use Deformable DETR / DINO instead.
- **Real-time inference** — heavy compared to YOLO.
- **Very few objects per image** — the fixed N=100 queries
  is overkill.

## Assumptions & caveats

- **Slow convergence** — DETR needs ~500 epochs; DAB-DETR /
  DINO speed this up.
- **Bipartite matching cost** — negative log class prob −
  L1 − GIoU. Tune the mix per dataset.
- **Positional encoding** — Cartesian or 2D sinusoidal.
- **N queries** — over-provision (N > max # objects).
- **"No-object" class** — allocated for unmatched queries.

## Related in this repo

- `iou-generalized-iou` — GIoU loss.
- `attention-mechanism`, `alibi-linear-attention-bias` —
  encoder/decoder building blocks.
- `panoptic-segmentation` — Mask2Former / MaskDINO extension.
- `yolo-object-detection`, `faster-rcnn-region-proposal` —
  compared alternatives.

## Run

```
python techniques/detr-transformer-detection/python/detr_transformer_detection.py
Rscript techniques/detr-transformer-detection/r/detr_transformer_detection.R
```

**Refs:** Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and Zagoruyko, S. "End-to-end object detection with transformers." In *ECCV*, 2020.

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
