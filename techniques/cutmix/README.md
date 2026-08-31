# CutMix (Reference Ch 30 Robustness)

Paste a **rectangular patch** of one training image onto another and mix
the labels in proportion to the patch's area. Yun et al. (2019) proposed
CutMix as a replacement for both mixup (which "ghosts" the mix pixel-wise)
and CutOut (which throws pixels away entirely).

## Algorithm

For each batch pair `(x_i, y_i), (x_j, y_j)`:

```
λ  ~  Beta(α, α)
(y1, y2, x1, x2)  =  rand_bbox(H, W, λ)
x̃  =  x_i with rectangle [y1:y2, x1:x2] replaced by x_j's rectangle
ỹ  =  λ_adj · y_i  +  (1 − λ_adj) · y_j
```

where `λ_adj = 1 − area(rect) / (H · W)` is the post-clipping adjusted
mix ratio.

## Why CutMix > mixup on images

- **Local pixel statistics are preserved** inside each rectangle;
  convnets like real (not blended) textures.
- **Full-image occlusion** style regularises against relying on a single
  patch — improves localisation and adversarial robustness.
- **Better weakly-supervised localisation** (from the paper's ImageNet
  experiments).

## When to use

- **Image classification** — CutMix is a default in modern recipes
  (EfficientNet-v2, DeiT, ConvNeXt) alongside mixup.
- **Detection / segmentation pretraining** — helps with localisation
  features via CutMix + strong augmentation.
- **Any convnet** where mixup gives modest gains.

## When NOT to use

- **Tabular / non-image data** — rectangular patches make no sense.
- **Very small images** (`≤ 16×16`) — a single rectangle can occupy
  most of the image; halve `α`.

## Files

- `python/cutmix.py` — from-scratch rectangular-bbox sampler
  (`rand_bbox` verbatim from Yun 2019), area-adjusted `λ`, softmax
  classifier trained on flattened tiny 16×16 synthetic images where
  each class has a bright square in a fixed corner. Vanilla + CutMix
  both reach clean accuracy 1.000; CutMix drops mean confidence
  0.996 → 0.898.
- `r/cutmix.R` — Keras / torch (R + Python); native R via array
  slicing in a custom `collate_fn`.

## Assumptions & caveats

- **α default** — `α = 1.0` (uniform `Beta(1, 1)`) is CutMix's default;
  `α = 0.2` for mixup.
- **Clipping matters** — always recompute `λ_adj` after bbox clipping;
  otherwise soft targets drift.
- **Combining CutMix + mixup** — many strong recipes sample which to
  apply per batch (torchvision `RandomChoice`).
- **Uniform label bounds** — for multi-label classification the target
  mix generalises to element-wise averaging of the binary label vector.
- **Not a replacement for adversarial defence** — CutMix helps calibration
  and small `L_p` robustness but is not certified.

## Related in this repo

- `mixup` — pixel-wise convex-combination variant.
- `label-smoothing` — label-side regulariser (no data mixing).
- `convolutional-nn`, `dropout-batchnorm`, `deep-mlp-backprop` — the
  underlying training loop.
- `data-augmentation` — the broader family (random crop, flip, colour
  jitter, RandAugment).

## Run

```
python techniques/cutmix/python/cutmix.py
Rscript techniques/cutmix/r/cutmix.R
```

**Refs:** Yun, S. et al. "CutMix: regularisation strategy to train strong classifiers with localizable features." *ICCV*, 2019; DeVries, T. & Taylor, G.W. "Improved regularisation of convolutional neural networks with cutout." 2017.

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
