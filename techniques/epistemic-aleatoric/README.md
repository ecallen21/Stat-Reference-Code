# Epistemic vs Aleatoric Uncertainty (Reference Ch 29 UQ)

Predictive uncertainty splits into two conceptually distinct pieces:

- **Aleatoric** — irreducible **data noise**; a property of the labelling
  process. Does not shrink with more data.
- **Epistemic** — reducible **model uncertainty**; shrinks with more data;
  grows away from the training support.

Kendall & Gal (2017) popularised the split for deep-learning vision;
Depeweg et al. (2018) formalised the information-theoretic decomposition.

## Regression decomposition

For an ensemble / posterior sample of Gaussian outputs `(μ_k, σ_k²)`:

```
Var_total(x) = 𝔼_k[ σ_k²(x) ]  +  Var_k[ μ_k(x) ]
                └── aleatoric ──┘   └── epistemic ──┘
```

## Classification decomposition (Depeweg 2018)

```
H_total(x) = H[ 𝔼_k p_k(y | x) ]
           = 𝔼_k H[ p_k(y | x) ]  +  I[ y ; θ | x, D ]
             └── expected entropy ─┘  └── mutual information ─┘
                    aleatoric              epistemic  =  BALD
```

The **mutual information** `I[y ; θ | x, D]` is Houlsby's BALD
acquisition function for active learning: prefer inputs where the models
disagree the most (high epistemic) rather than inputs where the label
itself is intrinsically noisy (high aleatoric).

## When each matters

- **Improving your model** — target regions where **epistemic** is high;
  more data / a bigger model can help.
- **Reporting a prediction interval** — sum both variances.
- **Deciding to gather labels** (active learning) — pick high **BALD**
  points, not just high entropy.
- **Rejecting a prediction** — abstain when epistemic > threshold; keep
  going when only aleatoric is high (the answer is just intrinsically noisy).

## Files

- `python/epistemic_aleatoric.py` —
  **Part 1**: 5-member deep ensemble with Gaussian-NLL head on
  heteroscedastic `sin(1.5x)` data. Explicit `(aleatoric, epistemic)`
  table across `x ∈ [-3, 3]`; in-dist epistemic sd 0.032 vs OOD 0.121
  (**3.8× larger**), while aleatoric follows the true `0.1 + 0.4|x|`
  noise function.
  **Part 2**: hand-set ensemble softmaxes illustrating the entropy /
  mutual-information split: agreement + certainty ⇒ MI = 0; agreement
  on an ambiguous class ⇒ MI = 0 (aleatoric only); disagreement
  ⇒ MI = 0.55 (epistemic).
- `r/epistemic_aleatoric.R` — `reticulate` + `uncertainty-toolbox` /
  `laplace-torch` / `pyro`.

## Assumptions & caveats

- **Aleatoric requires a variance head** or an explicit likelihood
  model; a point-prediction MSE loss cannot recover it.
- **Epistemic estimate quality depends on ensemble diversity** — a
  degenerate posterior (all members identical) reports zero epistemic
  even when uncertainty is high.
- **Cross-terms** — heteroscedastic aleatoric that itself depends on
  model uncertainty (rare) muddies the split.
- **Classification MI ≠ variance decomposition** — the two are related
  but not identical measures; report both when practical.
- **Not a substitute for calibration** — even a good split can be
  miscalibrated; combine with `calibration-scaling`.

## Related in this repo

- `deep-ensembles`, `mc-dropout`, `bayesian-neural-network`, `swag`,
  `last-layer-bayesian`, `evidential-deep-learning` — the ensemble /
  posterior sources for this decomposition.
- `active-learning` — BALD is `I[y ; θ | x, D]`.
- `selective-prediction` — abstain when epistemic is high.
- `covariate-shift-adaptation` — covariate shift = elevated epistemic.

## Run

```
python techniques/epistemic-aleatoric/python/epistemic_aleatoric.py
Rscript techniques/epistemic-aleatoric/r/epistemic_aleatoric.R
```

**Refs:** Kendall, A. & Gal, Y. "What uncertainties do we need in Bayesian deep learning for computer vision?" *NeurIPS*, 2017; Depeweg, S. et al. "Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning." *ICML*, 2018; Houlsby, N. et al. "Bayesian active learning for classification and preference learning (BALD)." 2011.

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
