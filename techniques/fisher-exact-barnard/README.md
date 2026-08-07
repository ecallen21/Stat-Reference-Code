# Fisher's Exact and Barnard's Exact Tests (Reference §8.4)

Both test independence in a 2×2 contingency table without the chi-square large-sample approximation. They differ in how the null distribution is constructed.

## Fisher's exact (conditional)

Condition on **both** row and column totals — the marginals are treated as ancillary. Under `H_0`, the (1,1) cell count is hypergeometric.

Two-sided p-value: sum the hypergeometric probabilities of all tables at least as extreme (probability-based rule, matching R's `fisher.test`).

## Barnard's exact (unconditional)

Condition only on the two row totals `(n_1, n_2)`. The common success probability `p` under `H_0` is a nuisance parameter — Barnard **maximizes** the p-value over `p ∈ [0, 1]` (Barnard 1945).

Uniformly more powerful than Fisher's on average (Berger & Boos 1994), but requires a supremum over a grid of `p` values (100–500 points is standard).

## Files

- `python/fisher_exact_barnard.py` — from-scratch Fisher's (hypergeometric p-value + odds-ratio) and Barnard's (Z-pooled statistic + grid supremum). On `[[8, 2], [3, 10]]`: Fisher's p = 0.0123 (matches `scipy.stats.fisher_exact`); Barnard's p ≈ 0.0081 (scipy 0.0108 — small differences from the choice of test statistic).
- `r/fisher_exact_barnard.R` — `stats::fisher.test` + `Exact::exact.test(method = "Z-pooled")`.

## When to use

- Any 2×2 table where either marginal is small (rule of thumb: expected cell < 5).
- Case-control designs with tight matching.
- Regulatory / auditing contexts where an exact p-value is required.

## Fisher vs Barnard

- **Fisher's** — universal, conservative, always available. Matched to R and the classic default in medical stats.
- **Barnard's** — modest power gains for small tables; recommended default in some modern references (Andres-Sanchez 1994; Mehta-Senchaudhuri 2003). Slightly slower.

## Assumptions & caveats

- **Independence of the two rows** (Barnard's) or **fixed marginals** (Fisher's).
- **Discreteness**: exact tests are inherently conservative — actual type-I error can be well below the nominal level.
- For larger tables (r × c), use `fisher.test` with `simulate.p.value = TRUE` or a Monte Carlo test.

## Run

```
python techniques/fisher-exact-barnard/python/fisher_exact_barnard.py
Rscript techniques/fisher-exact-barnard/r/fisher_exact_barnard.R
```

**Refs:** Fisher, R.A. "The logic of inductive inference." *J. R. Stat. Soc.* 98(1), 39–82, 1935; Barnard, G.A. "A new test for 2×2 tables." *Nature* 156(3954), 177, 1945; Berger, R.L. & Boos, D.D. "P values maximized over a confidence set for the nuisance parameter." *JASA* 89(427), 1012–1016, 1994.

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
