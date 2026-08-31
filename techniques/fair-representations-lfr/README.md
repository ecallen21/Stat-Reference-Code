# Learning Fair Representations (Reference Ch 31 Fairness)

**Learn a mapping `x → z` that keeps task-relevant information about
`y` but removes information about the protected attribute `A`.** Zemel
et al. (2013) — the original prototype-assignment LFR objective —
formalised the family; more recent variants (Bolukbasi 2016, Ravfogel
2020 INLP) use simple linear projections.

## Zemel 2013 objective

```
L  =  A_x · L_reconstruction  +  A_y · L_prediction  +  A_z · L_fairness

    L_recon    = ‖x − Σ_k P(z = k | x) M_k‖²
    L_predict  = BCE( y,  Σ_k P(z = k | x) w_k )
    L_fairness = ‖ P(z = k | A = 0) − P(z = k | A = 1) ‖_1
```

Alternating optimisation over prototypes `M_k` and per-example
assignment probabilities.

## Linear-projection cousin (Ravfogel 2020, INLP)

An easier-to-train first-order variant:

```
Repeat R times:
   v = unit direction that best predicts A from current z (logistic reg).
   z <- z − (z . v) v                (project onto v's null space)
Train the downstream classifier on z.
```

After enough iterations, no linear adversary can recover A from `z`;
Belrose 2023's *concept erasure* extends the idea to arbitrary linear
groups of directions.

## When to use

- **You need a REUSABLE FAIR ENCODING** shared across many downstream
  tasks (e.g. a corporate feature store) — Zemel-style prototypes are
  a compact fair intermediate.
- **You control the feature pipeline** and can strip A-info before
  handing data to third-party models.
- **You want *linear guardedness*** without full adversarial training —
  INLP is one linear regression per round.

## When NOT to use

- **You do not control features** — post-hoc processing is easier
  (`equalized-odds-postprocessing`).
- **You need a *proven* fairness guarantee** — INLP guards only against
  *linear* adversaries; a nonlinear head can still leak A.
- **Tiny feature spaces** — a few projection rounds can drain most of
  the useful signal (see caveat in the demo).

## Files

- `python/fair_representations_lfr.py` — self-contained **INLP-style
  iterated projection** demo. Synthetic data with two group-proxy
  features and four noise features. Results after 3 projection rounds:
  **raw features** — accuracy 0.844, DP ratio 0.40, linear adversary
  0.94; **debiased** — accuracy 0.53, DP ratio 0.90, linear adversary
  0.56.  Big fairness gain, real task-accuracy cost — the expected
  Pareto trade-off.
- `r/fair_representations_lfr.R` — `fairml` / `fairness` R packages;
  Python `aif360.algorithms.preprocessing.LFR` (Zemel prototype),
  `concept-erasure` / `inlp-oracle` (Ravfogel INLP).

## Assumptions & caveats

- **INLP guards only against linear adversaries** — a nonlinear classifier
  downstream may still expose A. Use adversarial training if you need
  stronger guarantees.
- **Number of rounds** — each round removes one direction; too many
  strip real signal.
- **Feature dimensionality** — INLP works best with rich, high-dim
  features; a 3-d input has only 3 directions to project out.
- **Zemel prototype LFR** is notoriously fragile to hyperparameters
  (weights `A_x, A_y, A_z`); prototypes tend to collapse.
- **Downstream metric** — LFR targets DP-style independence; equalised
  odds is not automatic.

## Related in this repo

- `reweighing-preprocessing`, `adversarial-debiasing`,
  `equalized-odds-postprocessing`, `exponentiated-gradient-reduction`
  — sibling mitigations.
- `dimensionality-reduction-pca` — the geometric intuition INLP shares.
- `counterfactual-fairness` — a fair-representation flavour based on
  causal graphs rather than linear projections.

## Run

```
python techniques/fair-representations-lfr/python/fair_representations_lfr.py
Rscript techniques/fair-representations-lfr/r/fair_representations_lfr.R
```

**Refs:** Zemel, R. et al. "Learning fair representations." *ICML*, 2013; Bolukbasi, T. et al. "Man is to computer programmer as woman is to homemaker? Debiasing word embeddings." *NeurIPS*, 2016; Ravfogel, S. et al. "Null it out: guarding protected attributes by iterative nullspace projection (INLP)." *ACL*, 2020; Belrose, N. et al. "LEACE: Perfect linear concept erasure in closed form." *NeurIPS*, 2023.

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
