# Independent Component Analysis (Reference §9.9)

**ICA** separates a multivariate signal `x = A s` into **statistically independent**, non-Gaussian source components. The canonical example is the cocktail-party problem: recover individual speakers from microphone mixtures.

## Model

```
x_i = A s_i                       observed mixture
s_i : independent, at most one Gaussian
```

- **Independence**, not just decorrelation. Correlated PCA components are only orthogonal.
- **Non-Gaussianity** is what makes independence identifiable. Any rotation of independent Gaussians remains independent Gaussian, so the model is undetermined for Gaussian sources.

## FastICA (Hyvärinen 1999)

1. **Center and whiten** `x` → `z` so `E[z zᵀ] = I`.
2. Find `w` such that `y = wᵀ z` has **maximal non-Gaussianity**, measured by negentropy:

```
J(y) ≈ (E[G(y)] − E[G(ν)])²         ν ~ N(0, 1)
```

3. **Fixed-point update**:

```
w ← E[z g(wᵀ z)] − E[g'(wᵀ z)] w             G = log cosh, g = tanh
```

4. **Symmetric decorrelation** across components (orthogonalize the `W` matrix).

## Files

- `python/independent_components.py` — from-scratch whitening + FastICA with logcosh nonlinearity + symmetric decorrelation. Demo on three mixed sources (sinusoid, square wave, Laplace noise): all recovered with `|corr| ≥ 0.998`; matches `sklearn.decomposition.FastICA`.
- `r/independent_components.R` — `fastICA::fastICA` (production).

## When to use

- **Blind source separation** — audio, EEG, fMRI, gene expression, financial returns.
- **Denoising** — separate a signal from an independent noise process.
- **Feature extraction** — ICA components as basis for downstream regression / classification when independent-source structure is plausible.

## PCA vs ICA

|                | PCA                                | ICA                             |
|----------------|------------------------------------|---------------------------------|
| Objective      | max variance                       | max independence (non-Gauss)    |
| Constraint     | orthogonal components              | statistically independent       |
| Requires       | Gaussian is fine                   | at most one Gaussian source     |
| Interpretation | directions of variance             | mixing of unknown sources       |

Both are typically preceded by whitening; ICA runs an extra rotation search after PCA-whitening.

## Assumptions & caveats

- **Component ordering** is arbitrary — ICA has no natural "importance" ranking like PCA's eigenvalues.
- **Signs / scales** are arbitrary — sources are recovered up to permutation, sign flip, and scaling.
- **Number of components** — typically fewer than the number of observed dimensions; PCA to `k` first, then ICA.
- **Local optima** — FastICA can converge to different fixed points across seeds; use multiple restarts or the deflation variant.

## Run

```
python techniques/independent-components/python/independent_components.py
Rscript techniques/independent-components/r/independent_components.R
```

**Refs:** Hyvärinen, A. "Fast and robust fixed-point algorithms for independent component analysis." *IEEE Trans. Neural Netw.* 10(3), 626–634, 1999; Hyvärinen, A., Karhunen, J. & Oja, E. *Independent Component Analysis*, Wiley, 2001.

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
