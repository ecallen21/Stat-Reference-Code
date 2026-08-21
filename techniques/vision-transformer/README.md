# Vision Transformer — ViT (Reference §27.x extra)

Dosovitskiy et al. (2021). Apply a plain transformer encoder to **image
patches** as if they were tokens. Beats CNNs at large data scale; matches
them at small data scale with augmentation / distillation (DeiT).

## Pipeline

1. **Patchify**: split `H × W` image into `P × P` patches → `N = HW / P²` flat vectors of size `P² · C`.
2. **Linear projection**: each patch → embedding of size `d_model`.
3. **[CLS] token + positional embedding**: prepend a learnable `[CLS]` token and add learned or sinusoidal positional embeddings to the `N + 1` tokens.
4. **Transformer encoder stack**: standard MHA + FFN + LayerNorm + residuals (see `transformer-encoder`).
5. **Classification head**: linear layer on the final `[CLS]` embedding.

## Variants

| Model | Key idea |
|---|---|
| **ViT-B / L / H** (Dosovitskiy 2021) | the original |
| **DeiT** (Touvron 2020) | distillation + strong augmentation for data efficiency |
| **Swin Transformer** (Liu 2021) | hierarchical windowed attention; captures multi-scale features |
| **BEiT / MAE / IJEPA** | masked-image pretraining objective |
| **DINO / DINOv2** | self-supervised, general-purpose visual features |
| **SAM** (Kirillov 2023) | segment-anything model; ViT + prompt encoder |
| **CLIP / SigLIP** | image-text contrastive pretraining; ViT + text tower |
| **MambaVision, ConvNeXt** | competitive non-transformer backbones |

## When to use

- **Large-scale image classification / retrieval / detection / segmentation**.
- **Multi-modal pretraining** — CLIP, SigLIP, DALL-E, LLaVA all use ViT backbones.
- **Foundation models for vision** — DINOv2, MAE, SAM as pretrained feature extractors.
- **When you have enough data** — ViT needs > ~1M images to beat well-tuned CNNs from scratch; smaller datasets benefit from DeiT-style distillation or transfer from a large pretrained model.

## Files

- `python/vision_transformer.py` — from-scratch ViT: `patchify` → linear patch embedding → `[CLS]` + positional embedding → 2 transformer encoder blocks → linear head. Demo on an 8×8×3 image with 4×4 patches (N=4, N+1=5 with `[CLS]`); output logits shape (3,) — 3 classes. Patchify sanity: `img[0:4, 0:4, :].ravel() == p[0]`.
- `r/vision_transformer.R` — `torch::nn_module` with `nn_conv2d(kernel_size=patch, stride=patch)` (equivalent to linear patch embed); `reticulate` + `torchvision.models.vit_b_16`, `timm.create_model('vit_base_patch16_224', pretrained=True)`.

## Assumptions & caveats

- **Patch size matters** — smaller patches (16×16 on 224² images = 196 tokens) give higher accuracy but quadratic compute.
- **Positional encoding** — learned absolute (original ViT), 2-D sinusoidal, or rotary (RoPE / RoPE-2D); all work; rotary extrapolates to larger resolutions.
- **Class token vs global average pool** — either works; GAP over patch tokens is slightly better in some benchmarks.
- **Pretraining scale is dominant** — ImageNet-1k is not enough for ViT-B to beat ResNet-50 without heavy augmentation; ImageNet-21k or JFT-300M gives ViT its edge.
- **Input resolution** — increasing resolution at fine-tune time (e.g. 224 → 384) improves accuracy; interpolate positional embeddings.
- **Attention quadratic in tokens** — high-res or 3-D data → use windowed / sparse attention (Swin, Perceiver).

## Related in this repo

- `transformer-encoder`, `attention-mechanism`, `embedding-layers` — the transformer building blocks reused here.
- `convolutional-nn` — the pre-ViT default for images; still competitive at small scale.
- `contrastive-learning`, `masked-language-modeling` — pretraining paradigms that scale ViT.
- `transfer-learning`, `knowledge-distillation` — how you actually use a pretrained ViT.

## Run

```
python techniques/vision-transformer/python/vision_transformer.py
Rscript techniques/vision-transformer/r/vision_transformer.R
```

**Refs:** Dosovitskiy, A. et al. "An image is worth 16×16 words: transformers for image recognition at scale." *ICLR*, 2021; Touvron, H. et al. "Training data-efficient image transformers & distillation through attention (DeiT)." *ICML*, 2021; Liu, Z. et al. "Swin Transformer: hierarchical vision transformer using shifted windows." *ICCV*, 2021; Oquab, M. et al. "DINOv2: learning robust visual features without supervision." *TMLR*, 2024.

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
