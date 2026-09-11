# ControlNet — Conditional Diffusion (Reference §47.213)

Zhang, Rao & Agrawala (2023, ICCV). Add a **trainable copy** of
the diffusion UNet's encoder that ingests an extra **condition**
(edge map, depth, pose skeleton) and injects its residuals into
the frozen base UNet via **zero-initialised convolutions**.

    frozen SD:      x → UNet → ε̂_base
    ControlNet:     (x, c) → Trainable_encoder → residuals
                    zero_conv(residuals) added into UNet skips
    Combined:       ε̂ = ε̂_base + zero_conv(residual)

Zero-init means training starts as identity to the base — no
regression on pretraining quality.

## Files

- `python/controlnet_conditional.py` — from-scratch toy
  ControlNet with linear layers:
  - **At init (W_out = 0)**: ‖combined − base‖ = 0 (exact identity).
  - **After 'training'**: condition steers score by ~8.4 units.
  - Residual magnitude scales monotonically with condition
    magnitude.
- `r/controlnet_conditional.R` — no R port; recommends
  `diffusers.ControlNetModel`, `lllyasviel/ControlNet`.

## When to use

- **Adding structural control to a pretrained diffusion model** —
  edge / depth / pose / segmentation.
- **When you can't afford to retrain the base model** — ControlNet
  fine-tunes only the new branch.
- **Multiple conditions** — stack multiple ControlNets, one per
  condition type.

## When NOT to use

- **Text-only conditioning** — the base model's text-cross-
  attention is enough.
- **Very short training budgets** — ControlNet needs ~thousands
  of (image, condition) pairs.
- **When lighter-weight adapters suffice** — T2I-Adapter or
  IP-Adapter are cheaper.

## Assumptions & caveats

- **Zero-init W_out** is critical for stable warm-up.
- **Frozen base** must be sufficient for the task — ControlNet
  can add structure but can't rescue a weak base.
- **Overfitting** — if the ControlNet dataset is small,
  regularisation (dropout, weight decay) is essential.
- **Multi-ControlNet** — mix multiple condition maps by summing
  their residuals with per-CN weights.

## Related in this repo

- `latent-diffusion-ldm`, `diffusion-model` — the base models
  ControlNet extends.
- `classifier-free-guidance` — orthogonal steering technique.
- `lora-peft` — parameter-efficient alternative fine-tuning.
- `t2i-adapter`, `ip-adapter` (in principle) — lighter-weight
  conditioning variants.

## Run

```
python techniques/controlnet-conditional/python/controlnet_conditional.py
Rscript techniques/controlnet-conditional/r/controlnet_conditional.R
```

**Refs:** Zhang, L., Rao, A. & Agrawala, M. "Adding conditional control to text-to-image diffusion models." *ICCV*, 2023; Mou, C. et al. "T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models." *AAAI*, 2024.

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
