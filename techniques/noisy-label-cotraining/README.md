# Co-Teaching for Noisy Labels (Reference §47.98)

Han, Yao, Yu, Niu, Xu, Hu, Tsang & Sugiyama (2018). Two networks
train on the same batch:

    1. Each ranks its OWN per-example losses.
    2. Each PEER keeps the SMALL-LOSS subset (fraction 1 − R(t)).
    3. Each net updates only on the PEER's kept subset.

Small-loss examples are more likely correctly labelled; peer
selection prevents the confirmation-bias loop of a single net
memorising its own errors.

## Files

- `python/noisy_label_cotraining.py` — simplified co-teaching
  demo with `sklearn` decision-tree base learners (noise-sensitive,
  unlike log-reg). Demo (n=800, 10 features, K=2 classes):
  - noise 0.10: baseline 0.733 → **co-teach 0.762** (+0.03)
  - noise 0.20: baseline 0.667 → **co-teach 0.710** (+0.04)
  - noise 0.40: baseline 0.550 → **co-teach 0.591** (+0.04)
  - noise 0.00: baseline slightly beats co-teach (throws away
    perfectly valid data).
- `r/noisy_label_cotraining.R` — no first-class R port;
  `cleanlab` (Python) handles noise detection via confident
  learning.

## When to use

- **Large-scale supervised learning with LABEL NOISE** — web-scraped
  labels, crowdsourced annotations, weak supervision.
- **Deep learning models** that overfit noisy labels.
- **Combined with re-labelling / cleanlab** for a full noise-
  handling pipeline.

## When NOT to use

- **Clean labels** — co-teaching wastes valid samples.
- **Robust base learners** (regularised linear) — noise less
  damaging; co-teaching hurts.
- **Small n** — throwing away examples has too high variance cost.
- **Highly correlated errors** across the two peers — both memorise
  the same noise.

## Assumptions & caveats

- **R(t) schedule** — start at 0 (keep all), ramp up to
  estimated noise rate; requires an estimate.
- **Base learners must differ** — otherwise peer selection is a
  no-op; use different seeds / architectures.
- **Noise model** — Co-teaching+ (Yu et al 2019) adds
  disagreement filtering for symmetric-flip noise.
- **Alternatives**: cleanlab / confident-learning, mixup with
  label smoothing, DivideMix.

## Related in this repo

- `label-smoothing`, `mixup`, `cutmix` — regularisation cousins.
- `active-learning-query-strategies`, `coreset-selection`,
  `stability-selection` — subset-selection methods.
- `deep-ensembles`, `mc-dropout`, `bayesian-neural-network`,
  `swag`, `conformal-classification`, `evidential-deep-learning`,
  `ood-detection`, `selective-prediction` — uncertainty and abstention.
- `catboost-ordered-boosting`, `bagging-oob` — robustness cousins.

## Run

```
python techniques/noisy-label-cotraining/python/noisy_label_cotraining.py
Rscript techniques/noisy-label-cotraining/r/noisy_label_cotraining.R
```

**Refs:** Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I. & Sugiyama, M. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS*, 2018; Northcutt, C.G., Jiang, L. & Chuang, I. "Confident learning: Estimating uncertainty in dataset labels." *JAIR* 70: 1373-1411, 2021.

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
