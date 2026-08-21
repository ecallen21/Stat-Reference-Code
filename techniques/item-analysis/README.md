# Item Analysis (Reference §22.2)

Classical Test Theory (CTT) item statistics for a K-item test scored 0/1.

## Difficulty (proportion correct)

```
p_j = fraction of examinees answering item j correctly
```

- Usable range: 0.2 – 0.8.
- Too easy (`p > 0.95`) or too hard (`p < 0.05`) → little information.

## Discrimination

Two common measures:

- **Upper-vs-lower** (Kelly 1939): `d = p_upper27% − p_lower27%`.
- **Point-biserial `r_pb`**: correlation of item score with **rest score** (total minus this item, avoiding part-whole bias).

Rule of thumb: `r_pb > 0.3` = good, `> 0.2` = acceptable, `< 0.15` = review.

## Distractor analysis (multiple-choice)

For each incorrect option: fraction choosing it + average total score among choosers. A **good** distractor is chosen more by low scorers than by high scorers.

## Files

- `python/item_analysis.py` — from-scratch difficulty + upper-lower discrimination + point-biserial (rest-score version) + auto flag. Demo (K = 10, item 3 has near-zero true discrimination): correctly flags item 3 as `low discrim` (r_pb ≈ 0).
- `r/item_analysis.R` — `psych::score.items`, `CTT::itemAnalysis`, `ShinyItemAnalysis`.

## When to use

- **Test development / QA** — screen new items before locking them into an operational form.
- **Post-hoc item review** — identify items to revise or drop.
- **Classroom assessment** — quick CTT diagnostics without IRT overhead.

## Cautions

- **Rest score** for `r_pb` avoids part-whole inflation; **corrected item-total correlation** in psych.
- **CTT vs IRT**: CTT is population-dependent; IRT (`rasch-model`, `two-three-pl-irt`) gives invariant item parameters.
- **Small samples** — flag statistics with CIs; a single p-hat isn't precise below n = 100.

## Run

```
python techniques/item-analysis/python/item_analysis.py
Rscript techniques/item-analysis/r/item_analysis.R
```

**Refs:** Ebel, R.L. & Frisbie, D.A. *Essentials of Educational Measurement*, 5th ed., Prentice Hall, 1991.

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
