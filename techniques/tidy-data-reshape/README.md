# Tidy Data + Long/Wide Reshape (Reference §41.15)

Wickham (2014). **Tidy data** rules:

1. Each variable is a column.
2. Each observation is a row.
3. Each type of observational unit is a table.

Reshape between forms:

- **Wide → long** (melt / `pivot_longer`) — one row per measurement,
  add id columns identifying which variable that row measures.
- **Long → wide** (pivot / `pivot_wider`) — one row per unit, one
  column per variable value.

Long form is preferred for tidyverse (ggplot2, lme4, tidymodels);
wide is more compact and matches classical tabular layouts.

## When to use

- **Repeated-measures / panel data** → long form for mixed models
  and time-series operations.
- **Presentation** → wide form for compact tables and heatmaps.

## When NOT to use

- **Data already in the right form** — reshape is a means, not an
  end.

## Files

- `python/tidy_data_reshape.py` — dictionary-based melt + pivot
  round-trip on a 3-patient × 3-week SBP dataset (custom, no
  pandas dependency).
- `r/tidy_data_reshape.R` — `tidyr::pivot_longer`/`pivot_wider`,
  `data.table::melt`/`dcast`, `reshape2::melt`/`dcast`/`acast` (R);
  `pandas.melt`/`pivot`/`stack`/`wide_to_long`, `polars.melt`/
  `pivot` (Python).

## Assumptions & caveats

- **Type consistency** — melting columns with different dtypes
  forces upcasting.
- **Duplicate keys** in long form need aggregation to pivot back
  (`aggfunc` / `values_fn`).
- **Names collisions** in wide form when levels contain reserved
  characters — quote or rename.

## Related in this repo

- `feature-engineering-time-series` (if present) — long-form
  operations.
- `sur-regression`, `panel-cointegration` — panel-shaped data.

## Run

```
python techniques/tidy-data-reshape/python/tidy_data_reshape.py
Rscript techniques/tidy-data-reshape/r/tidy_data_reshape.R
```

**Refs:** Wickham, H. "Tidy data." *Journal of Statistical Software*, 2014; Wickham, H. & Grolemund, G. *R for Data Science*, 2nd ed., O'Reilly, 2023.

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
