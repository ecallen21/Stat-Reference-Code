# Block Bootstrap for Dependent Data (Reference §10.4)

The IID bootstrap breaks the correlation structure of the data — for time series and spatially-correlated data it produces standard errors that are **too small**. The block bootstrap fixes this by resampling **blocks** of consecutive observations, preserving local dependence within each block.

## Variants

| Variant | Blocks | Note |
|---|---|---|
| **Non-overlapping** (Carlstein 1986) | disjoint partition into `⌈n/L⌉` blocks | Simplest; edge blocks may be truncated |
| **Moving blocks** (Kunsch 1989) | all overlapping length-`L` windows | The most common choice |
| **Circular** (Politis–Romano 1992) | as moving, but the series wraps around | Symmetric treatment of edge observations |

Each replicate concatenates `⌈n/L⌉` randomly drawn blocks and truncates to length `n`.

## Choosing L

Rule of thumb: **`L ≈ n^(1/3)`** for weakly-dependent series. Larger `L` preserves more autocorrelation; smaller `L` acts more like IID. Politis & White (2004) give a data-driven optimum via the autocorrelation function.

## Why it matters (from the demo)

On an AR(1) with φ = 0.7 and n = 300:

```
naive IID bootstrap SE(mean) = 0.077   ← WRONG (assumes independence)
moving-block   SE(mean) = 0.131         ← correctly ~1.7× larger
circular-block SE(mean) = 0.126
```

The IID bootstrap silently underestimates uncertainty by a factor of nearly 2 here — under the wrong dependence assumption, coverage of the resulting CI is far below nominal.

## Files

- `python/block_bootstrap.py` — moving and circular block bootstrap + `rule_of_thumb_block_length(n)`; demo compares block vs. IID SE on an AR(1). Optional cross-check against `arch.bootstrap.MovingBlockBootstrap`.
- `r/block_bootstrap.R` — from-scratch + `boot::tsboot`.

## Assumptions

- Data is stationary (or at least locally stationary within each block).
- Dependence has "short range" — decays over a length much less than the series length. For long-range dependence (fractional integration, unit roots), the block bootstrap doesn't apply cleanly.

## Run

```
python techniques/block-bootstrap/python/block_bootstrap.py
Rscript techniques/block-bootstrap/r/block_bootstrap.R
```

**Refs:** Künsch, H.R. "The jackknife and the bootstrap for general stationary observations." *Ann. Stat.* 17(3), 1217–1241, 1989; Politis, D.N. & Romano, J.P. "A circular block-resampling procedure for stationary data." *Explor. Limits Bootstrap*, Wiley, 1992; Politis, D.N. & White, H. "Automatic block-length selection for the dependent bootstrap." *Econ. Rev.* 23(1), 53–70, 2004.

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
