# Shannon Entropy (Reference §34.1)

Shannon (1948). Fundamental measure of uncertainty in a distribution.

## Formulas

```
Discrete:       H(p)     = − Σ_x p(x) log p(x)
Joint:          H(X, Y)  = − Σ_{x,y} p(x, y) log p(x, y)
Conditional:    H(Y | X) = H(X, Y) − H(X)
Chain rule:     H(X, Y)  = H(X) + H(Y | X)
Continuous:     h(p)     = − ∫ p(x) log p(x) dx      (differential entropy)
```

Units: **bits** (`log₂`), **nats** (`ln`), **Hartleys** (`log₁₀`).

## Continuous-data estimators

- **Histogram** with bin-width correction: `H_hist = H(bins) + log(width)`.
- **k-NN Kozachenko-Leonenko** (1987): unbiased, works in arbitrary
  dimension.
- **Kraskov-Stögbauer-Grassberger** (2004) for joint / MI.

## When to use

- **Anywhere uncertainty matters**: coding, compression, decision
  trees, cross-entropy loss, information-theoretic feature selection.
- **Measuring diversity** — biodiversity, category imbalance.
- **Baseline for other IT quantities** — MI = H(X) + H(Y) − H(X, Y).

## Files

- `python/shannon_entropy.py` — from-scratch discrete + histogram +
  Kozachenko-Leonenko k-NN estimator. Demo:
  - Coin flip: `H(fair) = 1.000`, `H(0.9/0.1) = 0.469`.
  - Joint / conditional identities hold (H(Y_copy | X) ≈ 0).
  - N(0,1) differential entropy: **true 1.4189, histogram 1.4218, k-NN
    (k=3) 1.4241**.
- `r/shannon_entropy.R` — `entropy`, `infotheo`, `FNN` (R); `scipy`,
  `NPEET` (Python).

## Assumptions & caveats

- **Discrete estimator is biased downward** for small samples;
  Hausser-Strimmer shrinkage helps.
- **Histogram estimator sensitive to bin choice** — use Freedman-
  Diaconis rule or CV.
- **k-NN estimator** — small `k` = variance; large `k` = bias.
- **Sample size for MI** — grows exponentially in dimension.

## Related in this repo

- `mutual-information` — the entropy-based association measure.
- `kl-divergence`, `cross-entropy-log-loss`, `f-divergences` — sibling
  info-theoretic quantities.
- `fisher-information`, `information-criteria`, `information-geometry`
  — info-theoretic model selection / geometry.
- `maximum-entropy` — the constrained-optimisation partner.

## Run

```
python techniques/shannon-entropy/python/shannon_entropy.py
Rscript techniques/shannon-entropy/r/shannon_entropy.R
```

**Refs:** Shannon, C.E. "A mathematical theory of communication." *Bell System Technical Journal*, 1948; Kozachenko, L.F. & Leonenko, N.N. "Sample estimate of the entropy of a random vector." *Problems of Information Transmission*, 1987; Cover, T.M. & Thomas, J.A. *Elements of Information Theory*, Wiley, 2006.

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
