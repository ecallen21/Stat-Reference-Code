# Textual Inversion (Reference §47.223)

Gal et al. (2022, ICLR 2023). Learn a new embedding vector v_* for
a placeholder token [S*] such that the FROZEN diffusion model
reconstructs 3-5 subject photos when prompted with [S*].

    θ (diffusion model) = frozen
    v_star (single token embedding, dim ~768) = trainable
    L = ‖ε_θ(x_subject, "A photo of [S*]", v_star) − ε‖²

~1000× cheaper than DreamBooth (only one embedding vector) but
less faithful for complex subjects.

## Files

- `python/textual_inversion.py` — toy TI on 4 subject "photos"
  (d=8 token embedding, 16-D output):
  - Loss 0.50 → 0.002 in 500 iters.
  - Cosine(v_learned, v_true) = **0.999**.
  - Parameters trained: 8 (TI) vs 512 (DreamBooth-style full
    fine-tune) — **64× cheaper**.
- `r/textual_inversion.R` — recommends `diffusers`
  `textual_inversion.py`, automatic1111.

## When to use

- **Fastest / cheapest personalisation** — one vector to train.
- **Style / concept transfer** — "in the style of [S*]".
- **Combining subjects** — mix multiple TI tokens per prompt.

## When NOT to use

- **When subject fidelity is critical** — DreamBooth / LoRA more
  faithful.
- **Complex subjects with unique geometry** — one embedding may
  underfit.
- **When the base LM's text encoder tokenises the subject as
  many sub-tokens** — TI trains only one vector.

## Assumptions & caveats

- **Embedding dim** matches CLIP text encoder (768 for SD1.5,
  1024 for SDXL).
- **Learning rate ~5e-4** typical.
- **Training photos** should be varied backgrounds; TI tends to
  overfit backgrounds.
- **Prompt template** matters — Gal 2022 uses templates like
  "a photo of [S*]" repeatedly at training.

## Related in this repo

- `dreambooth-subject-tuning` — more faithful, more expensive
  alternative.
- `lora-peft`, `prefix-prompt-tuning` — parameter-efficient
  fine-tuning cousins.
- `latent-diffusion-ldm`, `classifier-free-guidance` — SD
  pipeline neighbours.

## Run

```
python techniques/textual-inversion/python/textual_inversion.py
Rscript techniques/textual-inversion/r/textual_inversion.R
```

**Refs:** Gal, R. et al. "An image is worth one word: Personalizing text-to-image generation using textual inversion." *ICLR*, 2023.

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
