# Global Spatial Autocorrelation: Moran's I + Geary's C (Reference §23.3)

## Moran's I (Moran 1950)

```
I = (n / S_0) · Σ_i Σ_j W_ij (x_i − x̄)(x_j − x̄) / Σ_i (x_i − x̄)²
S_0 = Σ_ij W_ij
```

- `I ≈ +1` — strong positive spatial autocorrelation (clusters of similar values).
- `I ≈ 0` — random. `E[I] = −1/(n − 1)` under null.
- `I ≈ −1` — negative (checkerboard).

## Geary's C (Geary 1954)

```
C = ((n − 1) / (2 S_0)) · Σ_ij W_ij (x_i − x_j)² / Σ_i (x_i − x̄)²
```

- `C ≈ 0` — perfect positive.
- `C = 1` — random.
- `C > 1` — negative.

Geary's C emphasizes **local** differences; Moran's I emphasizes **overall** covariance. Both usually agree on presence of autocorrelation, disagree on magnitude.

## Files

- `python/morans_i_gearys_c.py` — from-scratch I + C + permutation p-value. Demo on 10×10 grid with additive `x + y` field: Moran's I = 0.90 (p = 0.001); Geary's C = 0.07. Random field: I = −0.16 (near null).
- `r/morans_i_gearys_c.R` — `spdep::moran.test`, `spdep::moran.mc`, `spdep::geary.test`.

## When to use

- **Diagnose spatial autocorrelation** before spatial regression.
- **Check residuals** of a spatial model — significant Moran's I on residuals ⇒ model misses spatial structure.
- **Cluster detection** (companion to `local-moran-lisa`).

## Assumptions & caveats

- **Depends on W** — different weights give different I; sensitivity check across contiguity / kNN / distance.
- **Analytic p-values** assume Normality; use permutation p when unsure.
- **Global measure** — hides local variation; complement with LISA.
- **Confounded with mean structure** — if trend not removed, I reflects the trend.

## Run

```
python techniques/morans-i-gearys-c/python/morans_i_gearys_c.py
Rscript techniques/morans-i-gearys-c/r/morans_i_gearys_c.R
```

**Refs:** Moran, P.A.P. "Notes on continuous stochastic phenomena." *Biometrika* 37(1/2), 17–23, 1950; Geary, R.C. "The contiguity ratio and statistical mapping." *Inc. Stat.* 5(3), 115–146, 1954.

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
