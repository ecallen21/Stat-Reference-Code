# Fieller CI for a Ratio (Reference §3.27)

Fieller (1954). Exact (under normality) CI for the ratio of two
normal means / regression slopes, robust to small denominators
(where the delta method silently fails).

## Formula

For `(θ̂₁, θ̂₂)` with covariance `V`:

    A r² − 2Br + C = 0
    A = θ̂₂² − z²·v₂₂
    B = θ̂₁·θ̂₂ − z²·v₁₂
    C = θ̂₁² − z²·v₁₁

If `A > 0` and `B² − AC > 0`:
`CI = ((B ± √(B² − AC)) / A)`; else CI is **unbounded** (denominator
too near 0).

## Files

- `python/fieller_interval_ratio.py` — Fieller + delta-method CI
  from scratch. Demo:
  - Strong denom (θ₂=2, v₂₂=0.1): r̂=2, Fieller (1.30, 3.09),
    delta (1.17, 2.83).
  - Weak denom (θ₂=0.5, v₂₂=0.3): Fieller (−∞, +∞) correctly
    signals non-identification; delta gives narrow (−9.2, 25.2).
- `r/fieller_interval_ratio.R` — `mratios`, `MBESS::ci.pi`,
  `drc::EDcomp` (R); from-scratch (Python).

## When to use

- **Ratio of two normal means** — bioassay potency, price ratios.
- **Cost-effectiveness ICER** — ratio of cost / effect differences.
- **IV / 2SLS β = Cov / Cov** — Anderson-Rubin ≈ Fieller.
- **Relative bioavailability** (log-scale ratios).

## When NOT to use

- **Denominator identically zero** — Fieller correctly returns
  unbounded; report descriptively.
- **Non-normal joint** — bootstrap the ratio directly.
- **High-dim ratios** — extend via multivariate delta / bootstrap.

## Assumptions & caveats

- **Joint normality of (θ̂₁, θ̂₂)** — Fieller is exact under this.
- **Known covariance V** — plug in the sandwich / model-based one;
  degrades if V is very noisy.
- **Empty CI** — occurs when the point estimate falls outside the
  parameter space; rare in practice but a diagnostic.
- **Unbounded CI** correctly represents that the ratio is not
  identified from the data — do not "fix" it with delta method.

## Related in this repo

- `cost-effectiveness-analysis` — natural application (ICER CIs).
- `delta-method`, `bootstrap-optimism-correction`,
  `nonparametric-bootstrap` — CI cousins.
- `weak-instruments-anderson-rubin` — IV analogue.

## Run

```
python techniques/fieller-interval-ratio/python/fieller_interval_ratio.py
Rscript techniques/fieller-interval-ratio/r/fieller_interval_ratio.R
```

**Refs:** Fieller, E.C. "Some problems in interval estimation." *JRSS-B*, 16(2): 175-185, 1954; Buonaccorsi, J.P. Fieller's Theorem chapter, *Encyclopedia of Biostatistics*, 2005.

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
