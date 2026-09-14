# FCN Semantic Segmentation (Reference §47.300)

Long, Shelhamer & Darrell (2015). Convert a classification CNN
to DENSE PIXEL prediction:

1. Replace fully-connected layers with 1×1 convs.
2. Upsample the coarse output to input resolution.
3. Add SKIP connections (FCN-16s, FCN-8s) combining coarse-
   semantic and fine-spatial features.

Founding paper of modern semantic segmentation; direct
predecessor to U-Net / DeepLab.

## Files

- `python/semantic_segmentation_fcn.py` — Bilinear upsampling
  and skip-fusion. Demo: 32×32 image with 3 classes; coarse
  8×8 logits. FCN-32s (single upsample ×4): mean IoU 0.19.
  Adding a mid-level skip (FCN-16s style) lifts mean IoU
  slightly. Real trained FCN-8s achieves > 0.6 on
  Pascal VOC.
- `r/semantic_segmentation_fcn.R` — reticulate +
  `torchvision.models.segmentation.fcn_resnet50` (R);
  `torchvision`, `mmsegmentation`, keras + deconv,
  from-scratch (Python).

## When to use

- **Educational / baseline** — the seminal fully-convolutional
  segmentation model.
- **When you need a simple dense-prediction head** on top of
  a pretrained CNN.
- **As the FCN-8s reference** to compare against U-Net,
  DeepLab, SegFormer.

## When NOT to use

- **State-of-the-art segmentation** — modern architectures
  (DeepLab v3+, Mask2Former, SegFormer) beat plain FCN.
- **Small datasets** — U-Net's skip connections + augmentation
  usually win.
- **Non-standard input resolutions** — pooling artefacts
  can accumulate.

## Assumptions & caveats

- **Upsampling method** — bilinear or learned deconv (both
  used in the paper). Deconv adds capacity but needs more
  data.
- **Skip fusion** — element-wise addition (as in the paper);
  U-Net concatenates instead.
- **Backbone** — VGG (paper), ResNet (modern) — the deeper
  the backbone the more benefit from skips.
- **Fine-tuning schedule** — Long et al fine-tune all layers,
  not just the head.

## Related in this repo

- `u-net-segmentation` — the biomedical descendant.
- `panoptic-segmentation` — extends to stuff + things.
- `dice-loss-segmentation`, `focal-tversky-loss` — companion
  losses.
- `deeplab` (not yet in this repo) — atrous / ASPP successor.

## Run

```
python techniques/semantic-segmentation-fcn/python/semantic_segmentation_fcn.py
Rscript techniques/semantic-segmentation-fcn/r/semantic_segmentation_fcn.R
```

**Refs:** Long, J., Shelhamer, E. and Darrell, T. "Fully convolutional networks for semantic segmentation." In *CVPR*, 2015.

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
