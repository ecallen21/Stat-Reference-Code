# Record Linkage / Entity Resolution (Reference §38.6)

Fellegi & Sunter (1969). Probabilistic matching of records across
databases without a shared unique identifier. For each candidate pair
build a comparison vector `γ` over shared fields, form the
likelihood-ratio

```
R(γ) = P(γ | match) / P(γ | non-match)
     = Π_i [ m_i^{γ_i} (1 − m_i)^{1 − γ_i}  /  u_i^{γ_i} (1 − u_i)^{1 − γ_i} ]
```

and classify by two thresholds:

- `R > T_hi` → **match**
- `R < T_lo` → **non-match**
- otherwise → **clerical review**

`(m_i, u_i)` can be domain-elicited or EM-estimated (Winkler 1988).

## When to use

- **Cross-database linkage** — EHR ↔ claims, census matches,
  registries without shared IDs.
- **Deduplication** within a single file.
- **Entity resolution** for names / addresses with typographical
  noise.

## When NOT to use

- **True unique identifier exists** — a deterministic join is
  correct and cheaper.
- **Very small overlap** — FS still fires but posterior match
  probabilities become unstable; consider full Bayesian linkage.

## Files

- `python/record_linkage.py` — Fellegi-Sunter log-weights, EM
  estimation of `(m, u, π)`, three-way classification. Demo: 500
  pairs (100 true matches, 400 non-matches). EM recovers `m ≈
  (0.97, 0.98, 1.00, 0.85)` vs truth `(0.95, 0.98, 0.99, 0.90)`;
  99/100 true matches classified correctly.
- `r/record_linkage.R` — `fastLink`, `RecordLinkage`, `reclin2` (R);
  `recordlinkage`, `dedupe`, `splink` (Python).

## Assumptions & caveats

- **Conditional independence** across fields — standard FS assumes
  fields agree independently within match/non-match; violated for
  correlated fields (address ↔ ZIP). Use log-linear models or
  Winkler's dependency-adjusted EM.
- **Blocking** — never compare all `N_A × N_B` pairs; block by
  cheap agreement fields (soundex, ZIP prefix) or LSH.
- **Threshold choice** balances false-match vs false-non-match
  rates. Report both alongside clerical review load.
- **String similarity** — extend `γ` from `{0, 1}` to graded
  similarity (Jaro-Winkler / edit distance) with binned weights.
- **Confidentiality** — PPRL (privacy-preserving record linkage)
  needed for cross-institution linkage of identifiers.

## Related in this repo

- `entity-linking` — text-side entity resolution (Wikipedia / KB).
- `fuzzy-matching` (if present) — string-similarity primitives.

## Run

```
python techniques/record-linkage/python/record_linkage.py
Rscript techniques/record-linkage/r/record_linkage.R
```

**Refs:** Fellegi, I.P. & Sunter, A.B. "A theory for record linkage." *JASA*, 1969; Winkler, W.E. "Using the EM algorithm for weight computation in the Fellegi-Sunter model of record linkage." *Proc ASA Survey Res Methods*, 1988; Herzog, T.N., Scheuren, F.J., & Winkler, W.E. *Data Quality and Record Linkage Techniques*, Springer, 2007.

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
