# Manski Bounds -- Partial Identification (Reference §15.34)

Manski (1990, 2003). When you refuse to invoke unconfoundedness,
exclusion, or monotonicity, you cannot **point-identify** the ATE.
What you can do is put **logical bounds** on it using only the data
and the outcome's support `[y_min, y_max]`.

## Worst-case (no-assumption) bounds

For bounded `Y ∈ [y_min, y_max]` and binary `T`:

    LB(E[Y(1)]) = E[Y|T=1]·P(T=1) + y_min·P(T=0)
    UB(E[Y(1)]) = E[Y|T=1]·P(T=1) + y_max·P(T=0)
    LB(ATE)     = LB(E[Y(1)]) − UB(E[Y(0)])
    UB(ATE)     = UB(E[Y(1)]) − LB(E[Y(0)])

**Width = y_max − y_min** — the bounds contain 0 by construction.

## Assumption-tightening add-ons

| Assumption | Meaning | Effect on bounds |
|---|---|---|
| **MTR** | `Y(1) ≥ Y(0)` for all units | ATE ≥ 0; LB tightens up |
| **MTS** | Treated select on higher `Y(t)` | E[Y|T=1] ≥ E[Y(1)]; UB tightens down |
| **MIV** | `E[Y(t) | V=v]` monotone in `v` | Uses an instrument to sharpen |

## Files

- `python/manski_bounds.py` — worst-case + MTR + MTS bounds. Demo
  (n=3000, Y∈[0,10], truth τ=2.0, positive confounding): naive OLS
  =+2.71 (biased upward); worst-case ATE ∈ [−3.64, +6.36] (width 10);
  MTR ATE ∈ [0, +6.36]; MTS UB(ATE) = +2.71 (= naive diff).
- `r/manski_bounds.R` — `bounds`, `relaxIV`, `RATest` (R);
  from-scratch (Python).

## When to use

- **Sensitivity to point-identifying assumptions** — "if we won't
  believe X, what can we still say?"
- **Uncontroversial reporting** — worst-case bounds are indisputable;
  they cannot be criticised for identification.
- **Combined with sensitivity analysis** — the width tells you how
  much of your conclusion rests on the assumption.

## When NOT to use

- **Bounds too wide to be interesting** — often the case with truly
  no assumptions; worst-case width = full outcome range.
- **Unbounded outcomes** — the method fails unless you trim / winsorise.
- **You have a defensible identification strategy** — point estimates
  are more informative when the assumptions are credible.

## Assumptions & caveats

- **Outcome support** — `y_min, y_max` must be known / defensible;
  wide support gives wide bounds.
- **MTR/MTS are also assumptions** — tightening comes at a cost;
  state them explicitly.
- **CI vs bounds** — Imbens-Manski CI covers the true point (not the
  identified set) at nominal rate.
- **Zero always in the bounds** — under no assumptions, you can never
  rule out "no effect".

## Related in this repo

- `sensitivity-e-value`, `rosenbaum-bounds` — parametric sensitivity
  alternatives to full partial identification.
- `iv-2sls`, `principal-stratification-cace` — point identification
  when a defensible instrument exists.
- `iptw`, `aipw-doubly-robust` — the point-identifying counterparts.

## Run

```
python techniques/manski-bounds/python/manski_bounds.py
Rscript techniques/manski-bounds/r/manski_bounds.R
```

**Refs:** Manski, C.F. "Nonparametric bounds on treatment effects." *American Economic Review*, 80(2): 319-323, 1990; Manski, C.F. *Partial Identification of Probability Distributions*, Springer, 2003.

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
