# Balanced Incomplete Block Design (BIBD) (Reference §32.9)

Yates (1936); Fisher (1935). A block design where each block sees
only a subset of the `v` treatments (block size `k < v`), but every
**pair** of treatments appears together in exactly `λ` blocks.

## Parameters

    r(k − 1) = λ(v − 1)         each treatment: r blocks × (k − 1) partners
    bk       = rv                total-plot identity

## Intra-block estimator

Adjusted treatment total `Q_i = T_i − (1/k) Σ_{blocks containing i} B_j`.
Estimate `τ̂_i = kQ_i / (λv)`. Pairwise contrasts have equal variance —
the design's key balance property.

## Files

- `python/balanced_incomplete_block_design.py` — Fano-plane BIBD
  (v=7, b=7, k=3, r=3, λ=1) constructor + intra-block estimator
  from scratch. Demo (true τ = (2, 1, 0, −1, −2, 1, −1)): recovers
  τ̂ ≈ (2.28, 0.94, −0.15, −0.87, −2.32, 0.95, −0.83).
- `r/balanced_incomplete_block_design.R` — `AlgDesign::optBlock`,
  `crossdes::isBIB / find.BIB`, `ibd`, `agricolae::design.bib`
  (R); pyDOE2, from-scratch (Python).

## When to use

- **More treatments than block capacity** — tasters, plots,
  clinical sessions can only rate `k` items at once but you have
  `v > k` treatments.
- **Balanced pairwise comparisons** — every treatment-pair has the
  same information — no favouritism.
- **Small experimental units** — sensory panels, animal studies,
  agricultural micro-plots.

## When NOT to use

- **All treatments fit in one block** — a complete-block design is
  simpler.
- **Adaptive / sequential settings** — BIBD assumes fixed
  pre-specified allocation.
- **Very small n** — BIBD requires `bk = rv` divisibility; may not
  exist for your `(v, k)`.

## Assumptions & caveats

- **Existence** — a BIBD with parameters `(v, b, r, k, λ)` needs
  the two identities to hold in integers; use `crossdes::find.BIB`
  to check.
- **Efficiency factor `E = λv / (rk) = (v − 1)/(v − k) · λ/r`** —
  reports intra-block information vs a full RCBD.
- **Inter-block information** can be recovered via combined
  analysis (mixed model with block random effect); increases
  efficiency when block-effect variance is small.
- **Randomisation** — allocate treatments to plots WITHIN each
  chosen block at random.

## Related in this repo

- `latin-square-design`, `crossover-design`, `split-plot-design`,
  `latin-hypercube-sampling` — DoE cousins.
- `linear-mixed-models`, `generalized-linear-mixed-models` — mixed-
  model fitters for BIBD data.
- `d-optimal-design`, `taguchi-methods` — algorithmic / robust
  design alternatives.

## Run

```
python techniques/balanced-incomplete-block-design/python/balanced_incomplete_block_design.py
Rscript techniques/balanced-incomplete-block-design/r/balanced_incomplete_block_design.R
```

**Refs:** Yates, F. "A new method of arranging variety trials involving a large number of varieties." *Journal of Agricultural Science*, 26: 424-455, 1936; Fisher, R.A. *The Design of Experiments*, Oliver & Boyd, 1935; Cochran, W.G. & Cox, G.M. *Experimental Designs*, 2nd ed., Wiley, 1957.

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
