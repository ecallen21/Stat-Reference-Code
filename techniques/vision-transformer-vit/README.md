# Vision Transformer — ViT (Reference §47.160)

Dosovitskiy et al. (2021). Applies a pure Transformer encoder to
sequences of flat image patches:

    1. Split image into P × P patches, linearly project → tokens.
    2. Prepend a [CLS] token + add learned position embeddings.
    3. L Transformer encoder blocks (multi-head self-attention + MLP).
    4. Linear head on the [CLS] token → class logits.

At sufficient scale (or with strong pre-training) matches or beats
CNNs on ImageNet. The convolutional inductive bias is traded for
expressive global attention over all patches at once.

## Files

- `python/vision_transformer_vit.py` — from-scratch single-block
  ViT with patchify + [CLS] token + position embeddings +
  single-head self-attention + MLP. Toy 8 × 8 shapes (horizontal
  stripes / vertical stripes / checkerboard, 3 classes, N = 180):
  - **LR on ViT [CLS] representation (random-init, frozen)**:
    test acc **1.000**
  - LR on flat 64-dim pixels: 1.000.
  - Both saturate on this easy task; the interesting property is
    that the [CLS] token summarises the whole 16-patch sequence
    via attention.
- `r/vision_transformer_vit.R` — no native R port; recommends
  `timm.vit_*`, `transformers.ViTModel`,
  `torchvision.models.vit_b_16` (all Python).

## When to use

- **Large-scale image classification** with abundant pretraining
  data (JFT-300M, ImageNet-21k).
- **Whenever attention over global patches** matters (medical
  imaging, satellite, ultra-high-resolution).
- **Multi-modal encoders** (ViT + text via CLIP / FLAVA).

## When NOT to use

- **Small datasets from scratch** — CNNs' inductive bias wins
  without pretraining.
- **Very high resolution without windowing** — quadratic attention
  costs O(T²); use Swin / hierarchical variants.
- **Latency-critical mobile inference** — MobileNet / EfficientNet
  win at similar accuracy per FLOP.

## Assumptions & caveats

- **Patch size** trades sequence length for information per token
  (16 × 16 on 224 × 224 → 196 tokens).
- **Position embeddings** — learned (paper) or 2-D sinusoidal;
  neither is essential with enough data.
- **[CLS] token** vs global average pooling — both work; [CLS]
  is the paper default.
- **Attention over all patches** scales as O(T²); Swin Transformer
  restricts to local windows for large images.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention`,
  `grouped-query-attention`, `mamba-state-space-transformer` —
  attention / sequence-model neighbours.
- `simclr-contrastive`, `barlow-twins` — SSL pretraining
  strategies commonly paired with ViTs.
- `gcn-graph-convolutional`, `graphsage-inductive-gnn`,
  `graph-attention-networks-gat` — attention on graphs sibling.

## Run

```
python techniques/vision-transformer-vit/python/vision_transformer_vit.py
Rscript techniques/vision-transformer-vit/r/vision_transformer_vit.R
```

**Refs:** Dosovitskiy, A. et al. "An image is worth 16×16 words: Transformers for image recognition at scale." *ICLR*, 2021; Touvron, H. et al. "Training data-efficient image transformers & distillation through attention." *ICML*, 2021; Liu, Z. et al. "Swin Transformer: Hierarchical vision transformer using shifted windows." *ICCV*, 2021.

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
