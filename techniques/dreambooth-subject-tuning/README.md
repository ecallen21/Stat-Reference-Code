# DreamBooth — Subject-Driven Fine-Tuning (Reference §47.222)

Ruiz, Li, Jampani, Pritch, Rubinstein & Aberman (2023, CVPR).
Personalise a pretrained text-to-image model with 3-5 photos of a
SUBJECT using a rare identifier token:

    prompt = "a photo of [V] dog"
    L_DB = L_recon(x_subject, "a [V] dog") + λ · L_prior("a dog")

**Prior-preservation loss** keeps the class ('dog') distribution
from drifting; identifier-token loss teaches the subject.

## Files

- `python/dreambooth_subject_tuning.py` — toy DreamBooth linear
  demo:
  - Subject-fit loss without vs with prior: both ~0.
  - Prior-preservation loss without prior = 0.358; **with prior
    = 0.004** (100× improvement) — model learns subject WITHOUT
    forgetting generic class.
- `r/dreambooth_subject_tuning.R` — recommends `diffusers`
  `train_dreambooth.py`, `kohya-ss/sd-scripts`.

## When to use

- **Custom subject** rendering (a specific pet, product, person).
- **Small-image budget** — 3-5 photos suffice.
- **Combined with LoRA** to reduce memory and enable stacking.

## When NOT to use

- **When Textual Inversion suffices** — cheaper and faster.
- **Very large subject datasets** — full fine-tune scales better.
- **Faces of real people without consent** — ethics + legal.

## Assumptions & caveats

- **Prior loss weight λ** ~ 1.0 typical.
- **[V] token** should be RARE in pretraining vocabulary (e.g.
  "sks", nonsense-string) to avoid interference.
- **~1000 steps** fine-tune enough for most subjects.
- **Overfitting to background** — augment or crop tightly.

## Related in this repo

- `textual-inversion` — cheaper single-token alternative.
- `lora-peft` — often paired with DreamBooth.
- `latent-diffusion-ldm`, `controlnet-conditional`,
  `classifier-free-guidance` — SD pipeline neighbours.

## Run

```
python techniques/dreambooth-subject-tuning/python/dreambooth_subject_tuning.py
Rscript techniques/dreambooth-subject-tuning/r/dreambooth_subject_tuning.R
```

**Refs:** Ruiz, N. et al. "DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation." *CVPR*, 2023.

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
