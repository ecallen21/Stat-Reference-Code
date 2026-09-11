# SAM — Segment Anything Model (Reference §47.229)

Kirillov et al. (2023, ICCV). Three components:

1. **Image encoder** (ViT-H) computes an image embedding.
2. **Prompt encoder** embeds point / box / mask / text prompts.
3. **Mask decoder** (light Transformer) predicts N masks +
   confidences from (image_embed, prompt_embed).

Trained on **1.1 B masks** in SA-1B. Zero-shot segments almost
anything with a click, box, or text prompt.

## Files

- `python/sam_segment_anything.py` — toy 3-blob scene + toy
  encoder / prompt encoder / decoder pipeline. Demonstrates the
  point / box prompt structure; real SAM uses a ViT-H backbone
  with cross-attention decoder.
- `r/sam_segment_anything.R` — recommends
  `facebookresearch/segment-anything`, SAM2, MobileSAM.

## When to use

- **Zero-shot segmentation** with minimal user input (a click).
- **Interactive annotation** tools.
- **Foundation-model** in a pipeline (SAM + LM for label
  propagation, SAM2 for video).

## When NOT to use

- **Domain shift** (medical imaging, satellite) — SAM was
  trained on natural images; fine-tune / MedSAM helps.
- **Latency-critical** small-device deployment — use MobileSAM.
- **Semantic labelling** — SAM gives shape masks, not class
  names.

## Assumptions & caveats

- **Image encoder is the bottleneck** — run once per image,
  cache the embedding for many prompts.
- **Multiple valid masks per click** — SAM returns 3 candidates
  (small / medium / large), pick by IoU predictions.
- **SA-1B licence** — non-commercial; check before shipping.
- **SAM2** (2024) extends to video with a memory bank.

## Related in this repo

- `vision-transformer-vit` — SAM's encoder backbone family.
- `dino-self-supervised-vision`,
  `mae-masked-autoencoders` — SSL vision cousins.
- `clip-vision-language` — commonly paired with SAM for
  text-driven segmentation (GroundedSAM).
- `gaussian-splatting-3d`, `nerf-neural-radiance-fields` —
  scene-understanding neighbours.

## Run

```
python techniques/sam-segment-anything/python/sam_segment_anything.py
Rscript techniques/sam-segment-anything/r/sam_segment_anything.R
```

**Refs:** Kirillov, A. et al. "Segment Anything." *ICCV*, 2023; Ravi, N. et al. "SAM 2: Segment Anything in Images and Videos." *arXiv:2408.00714*, 2024.

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
