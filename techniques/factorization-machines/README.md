# Factorization Machines (Reference §47.128)

Rendle (2010). Second-order model with shared low-rank factors for
pairwise interactions:

    y(x) = w₀ + Σᵢ wᵢxᵢ + Σ_{i<j} ⟨vᵢ, vⱼ⟩ xᵢxⱼ,   V ∈ ℝ^{p×k}

An identity collapses the pairwise sum:

    Σ_{i<j} ⟨vᵢ,vⱼ⟩xᵢxⱼ = ½ Σ_f ((Σᵢ v_{i,f} xᵢ)² − Σᵢ v_{i,f}² xᵢ²)

giving O(kp) per example instead of O(kp²). Great fit for sparse
categorical + numeric mixes (ad clicks, recsys).

## Files

- `python/factorization_machines.py` — from-scratch SGD FM
  (regression, squared loss). Demo (n=400, p=20 sparse binary,
  true positive interaction (0, 3) and negative (1, 5)):
  - **FM (k=4) MSE 0.026** vs Ridge MSE 0.035
  - Top learned interaction: **(0, 3)** positive; the (1, 5) is
    weaker but still ranks in the top pairs.
- `r/factorization_machines.R` — `libFMR`, `FactoRizationMachines`
  (R); `xLearn`, `pywFM`, `libFM`, from-scratch (Python).

## When to use

- **Sparse high-cardinality features** — categorical hashing spaces.
- **Recsys with side features** — hybrid MF + content.
- **Click-through-rate prediction** — one of the standard baselines.
- **When you want interaction structure + fast inference**.

## When NOT to use

- **Dense low-d features** — GBM / neural nets usually win.
- **Deep interactions** — DeepFM / xDeepFM / NCF.
- **Very small datasets** — regularised linear is simpler.

## Assumptions & caveats

- **λ tuning** — separate for w and V.
- **k (rank)** — 5–50 typical; more captures richer interactions.
- **Bias risk** with SGD; libFM defaults to MCMC or ALS.
- **Field-aware FM (FFM)** improves by giving each field its own
  factor.

## Related in this repo

- `matrix-factorization-als-recsys`,
  `neural-collaborative-filtering` — recsys cousins.
- `interaction-terms`, `friedmans-h-statistic`,
  `shap-interactions`, `pdp-ice-plots`,
  `ale-accumulated-local-effects` — interaction toolkits.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting`, `tabnet-tabular-attention` —
  tabular ML competitors.
- `target-encoding`, `feature-hashing`,
  `categorical-variable-coding` — categorical preprocessing.

## Run

```
python techniques/factorization-machines/python/factorization_machines.py
Rscript techniques/factorization-machines/r/factorization_machines.R
```

**Refs:** Rendle, S. "Factorization machines." *ICDM*, 2010.

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
