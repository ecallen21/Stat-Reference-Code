# CLIP — Contrastive Language-Image Pre-training (Reference §47.226)

Radford et al. (2021, ICML). Jointly train an image encoder E_i
and text encoder E_t on 400 M (image, caption) pairs with
**in-batch symmetric contrastive** loss:

    sim(i, t) = ⟨E_i(i) / ‖·‖, E_t(t) / ‖·‖⟩
    L = −log( exp(sim(i, t) / τ) / Σ_j exp(sim(i, j) / τ) )
        (symmetric: matched pairs on the diagonal)

Enables **zero-shot classification** via prompt engineering: compare
image embedding to embeddings of class-name prompts.

## Files

- `python/clip_vision_language.py` — toy CLIP with 5 aligned
  image-text pairs (d=32):
  - Similarity matrix: diagonal entries 0.93-0.97, off-diagonal
    near 0 (well-aligned).
  - **Contrastive loss with aligned diagonal: 0.0002** vs random
    pairing: 7.78.
  - Zero-shot classification: **5/5 correct**.
- `r/clip_vision_language.R` — recommends `openai/CLIP`,
  `open_clip`, `sentence-transformers`.

## When to use

- **Zero-shot image classification** — prompt with class names.
- **Vision-language retrieval** (image search by text).
- **Feature extractor** for many downstream tasks (SD text
  conditioning, RAG over images).

## When NOT to use

- **Fine-grained classification** — CLIP struggles vs specialists.
- **Small models** where alignment quality is poor.
- **Domains not in pretraining** (medical imaging, satellite).

## Assumptions & caveats

- **Temperature τ** learned; typical 0.01-0.1.
- **Prompt engineering** — "a photo of a {class}" beats bare
  class names by 1-3 pt.
- **Ensemble prompts** — average embeddings of 80+ templates
  (paper's ImageNet trick).
- **Data scale matters**: 400 M pairs; OpenCLIP-DataComp reached
  billions.

## Related in this repo

- `dense-passage-retrieval-dpr`, `colbert-late-interaction`,
  `sentence-transformers` (in principle) — retrieval neighbours.
- `contrastive-learning`, `simclr-contrastive`,
  `barlow-twins` — SSL contrastive cousins.
- `latent-diffusion-ldm` — text-encoder in SD.
- `sam-segment-anything`, `dino-self-supervised-vision` —
  foundation-model neighbours.

## Run

```
python techniques/clip-vision-language/python/clip_vision_language.py
Rscript techniques/clip-vision-language/r/clip_vision_language.R
```

**Refs:** Radford, A. et al. "Learning transferable visual models from natural language supervision (CLIP)." *ICML*, 2021.

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
