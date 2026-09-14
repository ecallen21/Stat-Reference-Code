# U-Net Segmentation (Reference §47.297)

Ronneberger, Fischer & Brox (2015). Encoder-decoder with SKIP
connections for pixel-level classification:

- **encoder**: successive 3×3 conv + downsample (contracts)
- **decoder**: transpose-conv upsample + concat with skip
- **output**: 1×1 conv to `n_classes` channels + softmax

Skip connections preserve high-resolution spatial detail
otherwise lost by downsampling. Standard for biomedical /
satellite / industrial-inspection segmentation.

## Files

- `python/u_net_segmentation.py` — 32×32 toy demo with a
  circular target region + additive noise. Simulated
  encoder-decoder with skip-connection concatenation → Dice
  score 0.59 vs the truth (real trained U-Net achieves > 0.95).
- `r/u_net_segmentation.R` — reticulate +
  `segmentation-models-pytorch` / MONAI (R); `smp.Unet`,
  `monai.networks.nets.UNet`, `keras-cv-attention-models`,
  from-scratch (Python).

## When to use

- **Biomedical / microscopy / MRI** — U-Net's founding domain;
  strong with limited data.
- **Satellite / aerial imagery** — landcover, buildings,
  roads.
- **When sharp boundaries matter** — skip connections retain
  edges.

## When NOT to use

- **Very large images without tiling** — encoder memory
  scales with image size; use tile-based inference.
- **Multi-scale detection** — U-Net gives dense masks, not
  boxes; use DETR / FCOS + segmentation head instead.
- **Very fine classes with small examples** — modern
  Transformer segmentation (Mask2Former, SegFormer) may beat
  vanilla U-Net.

## Assumptions & caveats

- **Loss choice** — Dice / Focal Tversky for imbalance;
  BCE + Dice hybrid is standard.
- **Data augmentation** — elastic deformation, flips, colour
  jitter — Ronneberger emphasises this for biomedical.
- **Padding** — "same" padding avoids boundary artefacts.
- **U-Net variants** — nnU-Net (Isensee 2021) is a strong
  self-configuring baseline.

## Related in this repo

- `semantic-segmentation-fcn` — FCN precursor.
- `dice-loss-segmentation`, `focal-tversky-loss` —
  companion losses.
- `panoptic-segmentation` — extension to stuff + things.
- `iou-generalized-iou` — evaluation metric.

## Run

```
python techniques/u-net-segmentation/python/u_net_segmentation.py
Rscript techniques/u-net-segmentation/r/u_net_segmentation.R
```

**Refs:** Ronneberger, O., Fischer, P. and Brox, T. "U-Net: Convolutional networks for biomedical image segmentation." In *MICCAI*, pp. 234-241, 2015.

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
