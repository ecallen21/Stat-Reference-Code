# Weight Standardization (Reference §47.172)

Qiao et al. (2019). Normalises the **weights** of each layer
instead of (or in addition to) the activations:

    Ŵ_{i, j} = (W_{i, j} − μ_i) / σ_i    (per output channel)

Combined with Group Normalization (Wu-He 2018), matches BatchNorm
at **micro-batch sizes** (1-4). Standard for object detection /
segmentation where large batches are infeasible.

## Files

- `python/weight_standardization.py` — 32-D features from a frozen
  random hidden layer with / without weight standardisation, LR
  head on top, on a 800-sample classification:
  - Plain features: 0.692
  - + BatchNorm: 0.662
  - + GroupNorm (small-batch fit): 0.679
  - + Weight Standardisation: 0.617
  - + WS + BN: 0.675
  - **+ WS + GN: 0.683** (recommended micro-batch combo).
  - The exact ordering depends on the head; the point is that WS
    tames per-channel weight scale so downstream normalisers
    behave well at any batch size.
- `r/weight_standardization.R` — pure-R weight-standardisation
  demo.

## When to use

- **Object detection / segmentation** where BN batch stats are
  unreliable (micro-batches per GPU).
- **BigGAN / self-supervised** setups where BN is discouraged.
- **Transfer learning** into small-batch regimes.

## When NOT to use

- **Large-batch training** where BN just works.
- **Very deep small networks** where the extra normalisation
  slows convergence with no visible generalisation gain.
- **Simple regression / logistic problems** — over-engineering.

## Assumptions & caveats

- **Per-output-channel stats** (row-wise for a linear layer,
  channel-wise for a conv).
- **Paired with GroupNorm** in practice (WS-GN); paired with LN
  in some Transformer-style variants.
- **ε ~ 1e-5** — smaller can destabilise.
- **Not free** — adds a mean-and-std op per layer per forward /
  backward.

## Related in this repo

- `dropout-batchnorm`, `spectral-normalization` — activation-
  normalisation neighbours.
- `rmsnorm-normalization` — Transformer-side sibling.
- `mixed-precision-training`, `gradient-checkpointing` — training
  toolbox neighbours.
- `label-smoothing`, `mixup`, `cutmix` — related generalisation
  regularisers.

## Run

```
python techniques/weight-standardization/python/weight_standardization.py
Rscript techniques/weight-standardization/r/weight_standardization.R
```

**Refs:** Qiao, S., Wang, H., Liu, C., Shen, W. & Yuille, A. "Weight standardization." *arXiv:1903.10520*, 2019; Wu, Y. & He, K. "Group normalization." *ECCV*, 2018.

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
