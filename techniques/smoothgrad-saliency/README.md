# SmoothGrad (Reference §47.115)

Smilkov, Thorat, Kim, Viégas & Wattenberg (2017). Averages the raw
saliency map over N Gaussian-perturbed copies of the input:

    Ŝ(x) = (1/N) Σᵢ |∇_x f(x + noise_i)|,   noise_i ~ N(0, σ² I).

Denoises noisy per-pixel gradients typical of ReLU / max-pool
networks. σ controls smoothness: larger σ = smoother, less
class-specific.

## Files

- `python/smoothgrad_saliency.py` — from-scratch SmoothGrad on
  a toy piecewise-linear-with-ReLU model. Demo (d=20, 5 signal
  features, N=100 perturbations):
  - σ=0.00 (raw grad): top-5 recall 0.40
  - σ=0.10: 0.40
  - σ=0.50: 0.20 (smearing)
  - σ=1.00: 0.40 (large-σ signal returns from denoising).
- `r/smoothgrad_saliency.R` — no first-class R port; `captum`
  (`NoiseTunnel`), `iNNvestigate` (Python).

## When to use

- **CNN / ReLU-network saliency** — visualise pixel importance
  without the "hairy" raw-gradient noise.
- **Post-hoc explanation** for a fixed classifier.
- **Complementary to Grad-CAM** at pixel resolution.
- **Model auditing** — verify a model uses "expected" regions.

## When NOT to use

- **Non-vision models** — SHAP / integrated gradients are more
  standard for tabular / text.
- **When faithfulness matters** — SmoothGrad is a smoother of raw
  gradients; not causally validated.
- **Very small models** where raw gradients are already clean.

## Assumptions & caveats

- **σ tuning** — 10–20 % of the input range is Smilkov's guidance.
- **N samples** — larger = smoother; 50-100 usually enough.
- **Symmetric noise** required — asymmetric perturbation biases the
  average.
- **Sign** — take `|∇|` for magnitude; consider signed VarGrad /
  VarianceGrad for stability.

## Related in this repo

- `integrated-gradients`, `grad-cam-saliency`, `shap-values`,
  `shap-interactions`, `lime-local-explanations`,
  `anchor-explanations`, `attribution-stability` — XAI toolkit.
- `randomized-smoothing`, `mc-dropout` — noise-averaging cousins
  in adversarial / uncertainty contexts.
- `neural-network-mlp`, `residual-connections`,
  `vision-transformer` — model families SmoothGrad extends to.

## Run

```
python techniques/smoothgrad-saliency/python/smoothgrad_saliency.py
Rscript techniques/smoothgrad-saliency/r/smoothgrad_saliency.R
```

**Refs:** Smilkov, D., Thorat, N., Kim, B., Viégas, F. & Wattenberg, M. "SmoothGrad: Removing noise by adding noise." *arXiv:1706.03825*, 2017.

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
