# VC (Vapnik-Chervonenkis) Dimension (Reference §46.11)

Vapnik & Chervonenkis (1971). Combinatorial measure of the capacity
of a binary function class `F`:

- A set `S = {x₁, …, x_d}` is **shattered** by `F` if every one of
  the `2ᵈ` labelings is realisable by some `f ∈ F`.
- `VC-dim(F) = largest d such that some set of size d is shattered.`

## Known dimensions

| Class | VC-dim |
|---|---|
| Half-lines in ℝ | 1 |
| Intervals in ℝ | 2 |
| Half-planes in ℝᵖ (with intercept) | `p + 1` |
| Axis-aligned rectangles in ℝ² | 4 |
| Decision stumps in ℝᵖ | `~ 2 log₂ p + 1` |
| Neural nets with W weights, L layers | `O(W · L)` |

## Sauer-Shelah lemma

For a class of VC-dim `d`, the growth function satisfies

    |F restricted to n points| ≤ (e·n / d)ᵈ

Combined with a uniform-convergence argument, this gives PAC bounds
of order `√(d · log n / n) + …`.

## Files

- `python/vc_dimension.py` — empirical shatter-based lower bound:
  for each candidate size k, try random sets and enumerate all `2ᵏ`
  labelings, testing whether each is linearly realisable via a
  least-squares surrogate. Demo: half-planes in ℝᵖ correctly recover
  the theoretical `p + 1` for p ∈ {1, 2, 3, 5}. Also prints
  Sauer-Shelah bounds for n = 100 and d = 1, 3, 5, 10.
- `r/vc_dimension.R` — no CRAN package; describes the from-scratch
  approach and related SVM-radius/margin diagnostics (`e1071`).

## When to use

- **Theoretical model capacity** — proving PAC learnability.
- **Class comparison** — bigger VC-dim ⇒ potentially needs more
  samples for the same guarantee.
- **Selecting complexity penalties** — structural risk minimisation
  scales its penalty with `√(d/n)`.

## When NOT to use

- **Non-binary classes / regression** — VC-dim is defined for binary
  functions; use **fat-shattering**, **Natarajan**, or
  **pseudo-dimension**.
- **Data-dependent capacity** — Rademacher complexity is
  distribution-dependent and typically tighter.
- **Practical validation** — CV / hold-out beats VC for empirical
  model selection.

## Assumptions & caveats

- **Distribution-free** — the bound applies to ANY iid distribution;
  can be loose for benign inputs.
- **Empirical estimator noise** — the shatter check is combinatorial
  and expensive (`2ᵏ` labels per set); the lstsq surrogate can
  miss shatterings near the boundary → estimate is a **lower bound**.
- **Infinite VC** — some classes (all measurable functions, some
  kernel-based) have infinite VC; use fat-shattering.

## Related in this repo

- `rademacher-complexity` — distribution-dependent capacity measure.
- `efron-stein-inequality` — variance bounds used in PAC proofs.
- `svm-classifier` — a class whose VC-dim controls its
  generalisation.

## Run

```
python techniques/vc-dimension/python/vc_dimension.py
Rscript techniques/vc-dimension/r/vc_dimension.R
```

**Refs:** Vapnik, V.N. & Chervonenkis, A.Ya. "On the uniform convergence of relative frequencies of events to their probabilities." *Theory of Probability & Its Applications*, 16(2): 264-280, 1971; Vapnik, V.N. *Statistical Learning Theory*, Wiley, 1998.

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
