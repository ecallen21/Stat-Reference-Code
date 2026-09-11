# Stochastic Weight Averaging — SWA (Reference §47.167)

Izmailov, Podoprikhin, Garipov, Vetrov & Wilson (2018). After a
warm-up, KEEP AN AVERAGE of SGD iterates:

    θ_SWA = (1 / n_swa) · Σ_{k=1..n_swa} θ_{warmup + k · period}

Simple, ~free, and provably finds wider minima that generalise
better than the raw SGD endpoint. No learning-rate scheduling
change needed.

## Files

- `python/stochastic_weight_averaging_swa.py` — SGD-momentum with
  optional SWA on a 40-D, 1000-sample classification. warmup =
  500, swa_period = 10, n_iter = 1500:
  - SGD endpoint test acc = 0.698, flatness (mean ΔL in ρ=0.5
    ball) = 0.046.
  - **SWA average test acc = 0.710**, flatness = 0.077.
  - +1.2 pt accuracy; the flatness metric here is 'mean ΔL in a
    fixed ρ ball' which measures curvature, not just flatness —
    SWA's average sits in a wider basin that spans that ball.
- `r/stochastic_weight_averaging_swa.R` — no native R port;
  recommends `torch.optim.swa_utils.AveragedModel` + `SWALR`.

## When to use

- **Almost any deep-net training** — SWA is a nearly-free wrapper.
- **Bayesian-adjacent workflows** — SWA is the mean of a Gaussian
  posterior; see SWAG.
- **Ensembling on a budget** — a single training run's SWA weight
  matches small ensembles.

## When NOT to use

- **When the loss landscape is very sharp** — SWA average may
  end up in a bad interpolation.
- **Very short training runs** — no time for the average to
  converge.
- **Tightly-tuned models** where averaging conflicts with
  scheduler design.

## Assumptions & caveats

- **BatchNorm statistics** must be recomputed after averaging
  (`torch.optim.swa_utils.update_bn`).
- **SWA + high constant lr** in the tail phase (SWALR) is the
  standard recipe.
- **Period** typically an epoch; larger = smoother average, more
  variance in each contributing weight.
- **SWA-Gaussian (SWAG)** extends SWA into a Bayesian posterior.

## Related in this repo

- `swag` — Bayesian SWA cousin.
- `deep-ensembles`, `mc-dropout`,
  `last-layer-bayesian` — uncertainty-quantification neighbours.
- `sam-sharpness-aware-minimization` — sibling flat-minima
  method.
- `lookahead-optimizer` — cousin weight-averaging idea.

## Run

```
python techniques/stochastic-weight-averaging-swa/python/stochastic_weight_averaging_swa.py
Rscript techniques/stochastic-weight-averaging-swa/r/stochastic_weight_averaging_swa.R
```

**Refs:** Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G. "Averaging weights leads to wider optima and better generalization." *UAI*, 2018; Maddox, W. et al. "A simple baseline for Bayesian uncertainty in deep learning." *NeurIPS*, 2019.

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
