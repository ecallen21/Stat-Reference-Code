# Multi-Vari Charts (Reference §37.12)

Seder (1950). Visualise multiple sources of variation in a single plot
so the eye — and a matching variance decomposition — pick out which
layer dominates BEFORE investing in DOE.

## Three families of variation

- **Within-piece** (positional) — multiple measurements on the same
  unit.
- **Between-piece** (piece-to-piece) — units within the same time
  window.
- **Temporal** (time-to-time) — shift-to-shift, day-to-day.

## Decomposition

For each time `t` and piece `p`:

```
within_var(t,p)  = Var(measurements on piece p at time t)
piece_mean(t,p)  = mean of those measurements
between_piece(t) = Var over p of piece_mean(t,·)
between_time     = Var over t of mean_p(piece_mean(t,·))
```

Report each contribution as a share of total variance.

## When to use

- **Screening step** before DOE — identify which nested layer of
  variation dominates and target improvement effort there.
- **Manufacturing / process characterisation** with a natural
  time × piece × position nesting.

## When NOT to use

- **Single-source variation** — a plain histogram or control chart is
  simpler.
- **Formal variance-components model** — use `variance-components`
  (REML) for hierarchical inference.

## Files

- `python/multi_vari_charts.py` — variance decomposition on 5 × 4 × 3
  synthetic data. Demo: **within = 7.7 %, between-piece = 39.1 %,
  between-time = 53.2 %** — temporal source dominates.
- `r/multi_vari_charts.R` — `SixSigma::ss.ci` (R reference); custom
  (Python).

## Assumptions & caveats

- **Balanced design assumed** in the compact decomposition; unbalanced
  data needs a mixed-model REML.
- **Additive layers** — assumes no interaction between time and
  piece.
- **Chart interpretation is heuristic** — pair with a formal ANOVA /
  REML for inference.

## Related in this repo

- `variance-components` — formal REML decomposition.
- `nested-random-effects` — nested-model estimation.
- `shewhart-control-charts`, `cusum-charts`, `ewma-charts` — SPC
  cousins.

## Run

```
python techniques/multi-vari-charts/python/multi_vari_charts.py
Rscript techniques/multi-vari-charts/r/multi_vari_charts.R
```

**Refs:** Seder, L.A. "Diagnosis with diagrams." *Industrial Quality Control*, 1950; Montgomery, D.C. *Introduction to Statistical Quality Control*, 7th ed., Wiley, 2013.

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
