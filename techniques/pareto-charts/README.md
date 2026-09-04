# Pareto Charts (Reference §37.14)

Juran (1954). "The vital few and the trivial many." Sort defect
categories by frequency (or cost) descending and overlay a cumulative-
percent line — the small set of categories left of the ~80 % mark is
where improvement effort belongs.

## Method

1. Tally counts (or cost) per category.
2. Sort descending.
3. Compute `pct = count / total` and `cum_pct` cumulatively.
4. Report the smallest set of categories whose `cum_pct` reaches a
   threshold (typically 80 %).

## When to use

- **Root-cause prioritisation** after 5-Whys / Ishikawa.
- **DOE factor screening** — focus on the vital few.
- **Cost-of-quality** analysis with `count → cost weight`.

## When NOT to use

- **Uniform-cause processes** — Pareto adds no signal when all causes
  are similar magnitude.
- **Continuous outcomes** — a histogram or run chart fits better.

## Files

- `python/pareto_charts.py` — sorted table + cumulative % + vital-
  few identification. Demo (10 defect types, N=401): **4 categories
  (Solder, Wire, Coating, Alignment) cover ~83 %** of defects.
- `r/pareto_charts.R` — `qcc::pareto.chart` (R reference); custom
  (Python).

## Assumptions & caveats

- **Categories are mutually exclusive** — double-counting inflates
  Pareto rank of overlapping causes.
- **Frequency ≠ importance** — weight by cost or severity when
  cheap-and-frequent competes with rare-and-catastrophic.
- **80/20 is a heuristic**, not a law — real ratios vary; use the
  cumulative line, not a fixed number.
- **Stable process assumed** — a shifting process makes yesterday's
  Pareto obsolete tomorrow.

## Related in this repo

- `shewhart-control-charts` — establishes stability first.
- `multi-vari-charts` — variation-source screening (cousin to
  Pareto).
- `fishbone-diagrams` (if present) — cause-and-effect input to
  Pareto.

## Run

```
python techniques/pareto-charts/python/pareto_charts.py
Rscript techniques/pareto-charts/r/pareto_charts.R
```

**Refs:** Juran, J.M. "Universals in management planning and controlling." *Management Review*, 1954; Montgomery, D.C. *Introduction to Statistical Quality Control*, 7th ed., Wiley, 2013.

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
