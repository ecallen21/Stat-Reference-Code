# Independent Component Analysis (Reference §25.1)

Hyvärinen & Oja (2000). Given mixed observations `X = A S` with
**statistically independent** non-Gaussian sources `S` and unknown
mixing matrix `A`, ICA recovers `S` up to permutation + scale by
**maximising non-Gaussianity** of the recovered signals (central-limit
intuition: sums of independents are more Gaussian).

## FastICA (per-component)

1. **Whiten** `X`: `Z = V (X − mean)` so `cov(Z) = I`.
2. Iterate `w ← 𝔼[Z · g(wᵀZ)] − 𝔼[g'(wᵀZ)] w` with a nonlinearity
   `g(u) = tanh(u)` or `g(u) = u exp(−u²/2)`.
3. **Deflate** — orthogonalise against previous components.

## When to use

- **Blind source separation** — EEG, MEG, audio (cocktail-party).
- **Feature extraction** where independence is a scientific prior.
- **Denoising** — identify and subtract a component matching an
  artefact.

## When NOT to use

- **Gaussian sources** — ICA cannot separate them; the mixing is
  identifiable only up to an orthogonal rotation.
- **Highly correlated / non-independent sources** — the assumption
  fails.
- **Large mixing residual noise** — ICA is essentially noise-free.

## Files

- `python/ica.py` — from-scratch whitening + FastICA per-component
  update with tanh nonlinearity. Demo mixes a sine wave and a square
  wave through a 2×2 matrix; recovered sources correlate |0.999| with
  the true sources.
- `r/ica.R` — `fastICA`, `ica`, `Rica` (R); `sklearn.decomposition
  .FastICA`, `picard`, `MNE-Python` (Python).

## Assumptions & caveats

- **Sign / permutation ambiguity** — inherent; report `|corr|`.
- **Scale ambiguity** — sources are recovered up to a scalar.
- **Number of components** — usually equal to `d`; robust to
  `n_components < d` when signals are compressible.
- **Non-Gaussian assumption** — check via kurtosis / entropy of
  recovered components.
- **Nonlinearity choice** — `tanh` for symmetric, `u³` for
  super-Gaussian, `−exp(−u²/2)` for sub-Gaussian.
- **Sensitive to outliers** — robust ICA variants exist (`fastICA`
  with M-estimators).

## Related in this repo

- `pca` (if present) / `dimensionality-reduction-pca` — variance-based
  alternative.
- `nmf` — non-negativity-constrained sibling.
- `sparse-pca`, `dictionary-learning`, `variational-autoencoder` —
  other latent-factor techniques.
- `canonical-correlation` — cross-modality dependence sibling.
- `blind-source-separation` (if present).

## Run

```
python techniques/ica/python/ica.py
Rscript techniques/ica/r/ica.R
```

**Refs:** Hyvärinen, A. & Oja, E. "Independent component analysis: algorithms and applications." *Neural Networks*, 2000; Comon, P. "Independent component analysis, a new concept?" *Signal Processing*, 1994.

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
