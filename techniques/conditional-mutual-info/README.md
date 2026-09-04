# Conditional Mutual Information + CI Test (Reference §34.13)

**`I(X; Y | Z)`** — mutual information between X and Y after
controlling for Z. Zero iff `X ⊥⊥ Y | Z`.

## Formula

```
I(X; Y | Z)  =  H(X, Z) + H(Y, Z) − H(X, Y, Z) − H(Z).
```

## CMI-based conditional-independence test

Under H0 (`X ⊥⊥ Y | Z`), permuting Y **within strata of Z** leaves
the joint (X, Y, Z) distribution unchanged. Compare observed CMI to
permutation null.

## When to use

- **Causal discovery** — CI oracle for PC / FCI algorithms.
- **Confounder detection** — does conditioning on Z remove
  X-Y dependence?
- **Feature-selection** — CMI-based feature scoring (mRMR).

## When NOT to use

- **Continuous data without a good estimator** — KSG / kernel-based
  needed; discrete demo suffices for small support.
- **High-dim Z** — CMI is data-hungry; the curse of dimensionality
  bites hard.

## Files

- `python/conditional_mutual_info.py` — discrete CMI via entropy
  decomposition + permutation test that preserves Z-strata. Demo:
  - **(a) X ⊥⊥ Y | Z** (Z generates both): `I(X;Y|Z) = 0.0001`,
    p = 0.895 (accept CI); unconditional `I(X;Y) = 0.40` (spurious).
  - **(b) X → Y direct**: `I(X;Y|Z) = 0.406`, p = 0.000 (reject CI).
- `r/conditional_mutual_info.R` — `bnlearn::ci.test`, `condMI` (R);
  `NPEET`, `causal-learn.CIT` (Python).

## Assumptions & caveats

- **Permutation must preserve Z** — permuting Y globally destroys the
  test.
- **Small strata** — sparse Z levels give unstable CMI estimates.
- **Bias correction** — Miller-Madow, Chao-Shen for small samples.
- **Continuous variables** — Runge 2018 k-NN CMI test; Zhang 2011
  kernel-based CI (KCI).
- **Multiple testing** across many (X, Y, Z) triples — FDR control.

## Related in this repo

- `mutual-information`, `shannon-entropy`, `kl-divergence`,
  `transfer-entropy` — sibling info-theoretic quantities.
- `dag-inference`, `pc-algorithm` (if present) — causal-discovery
  cousins.
- `chi-square-tests`, `partial-correlation` — classical CI tests.

## Run

```
python techniques/conditional-mutual-info/python/conditional_mutual_info.py
Rscript techniques/conditional-mutual-info/r/conditional_mutual_info.R
```

**Refs:** Runge, J. "Conditional independence testing based on a nearest-neighbour estimator of conditional mutual information." *AISTATS*, 2018; Zhang, K. et al. "Kernel-based conditional independence test and application in causal discovery." *UAI*, 2011.

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
