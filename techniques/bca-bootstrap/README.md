# BCa Bootstrap CI + Comparison of Bootstrap CI Methods (Reference §10.3; also covers §10.14)

**BCa** = **B**ias-**C**orrected and **a**ccelerated. Two adjustments turn the plain percentile CI into a second-order-accurate one:

- `z₀` (bias correction) — the fraction of bootstrap replicates below `θ̂`, mapped to a z-value:
  ```
  z₀ = Φ⁻¹( fraction of θ* below θ̂ )
  ```
- `a` (acceleration) — measures skewness of the sampling distribution; estimated from jackknife deviations of `θ̂`:
  ```
  a = Σ (J̄ − J_i)³  /  (6 · (Σ (J̄ − J_i)²)^(3/2))
      J_i  =  θ̂ computed on the sample with obs i removed.
  ```

Adjusted percentiles for the `α`-level CI:

```
α₁ = Φ( z₀ + (z₀ + z_{α/2})   / (1 − a(z₀ + z_{α/2})) )
α₂ = Φ( z₀ + (z₀ + z_{1−α/2}) / (1 − a(z₀ + z_{1−α/2})) )

CI  =  [ Q_{α₁}(θ*),  Q_{α₂}(θ*) ]
```

Plain percentile is a special case (`z₀ = 0`, `a = 0`).

## Also covers §10.14 — comparison of CI methods

`compare_ci_methods()` computes **all four** common bootstrap CIs on the *same* set of replicates for a fair comparison:

| CI | Best behavior | Coverage / order |
|---|---|---|
| **Percentile** | Simplest; monotone transformation-respecting | 1st-order accurate |
| **Basic (pivotal)** | Corrects bias in `θ̂` via reflection | 1st-order accurate |
| **Normal** | Assumes Gaussian sampling distribution | 1st-order accurate |
| **BCa** | Handles bias + skewness | **2nd-order accurate** |

Wider isn't worse — a narrower "normal" CI is often narrower because it's *wrong*, not because it's tighter for the right reason. Prefer BCa when in doubt.

## Files

- `python/bca_bootstrap.py` — from-scratch BCa (with z₀ + jackknife-based acceleration) + `compare_ci_methods()` that reports all four intervals on shared replicates. Percentile and basic CIs match `scipy.stats.bootstrap` to 12 dp; BCa is close (scipy's acceleration handles near-symmetric samples slightly differently).
- `r/bca_bootstrap.R` — from-scratch + `boot::boot.ci(type=c("perc","basic","bca","norm"))`.

## Assumptions

- Same as the plain bootstrap: independent observations; enough distinct values to make resampling meaningful.
- The jackknife-based `a` becomes noisy for tiny `n`; for `n < 20` BCa can behave erratically. Use percentile in that regime.

## Run

```
python techniques/bca-bootstrap/python/bca_bootstrap.py
Rscript techniques/bca-bootstrap/r/bca_bootstrap.R
```

**Refs:** Efron, B. "Better bootstrap confidence intervals." *JASA* 82(397), 171–185, 1987; DiCiccio, T.J. & Efron, B. "Bootstrap confidence intervals." *Stat. Sci.* 11(3), 189–228, 1996; Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and Their Application*, Cambridge, 1997 (Ch. 5).

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
