# Quantitative Content Analysis (Reference §42.17)

Krippendorff (2019), Neuendorf (2017). Systematic manual coding of
text into categories with an explicit codebook and reliability
assessment. Five stages:

1. **Unitize** — define the coding unit (word, sentence, doc).
2. **Sample** — draw a representative subset.
3. **Code** — two or more human coders apply the codebook.
4. **Reliability** — Krippendorff α on a double-coded subset (or
   Cohen κ for 2 raters, categorical); target α ≥ 0.80.
5. **Adjudicate** — reconcile disagreements before analysis.

## Reliability thresholds (Krippendorff 2019)

| α | Interpretation |
|---|---|
| ≥ 0.80 | good |
| 0.667 – 0.80 | tentative |
| < 0.667 | unreliable |

## When to use

- **Qualitative → quantitative** conversion of text for hypothesis
  testing.
- **Small corpora** where full annotation is feasible and ML would
  be under-powered.
- **Sensitive domains** (medical, legal) where model outputs need
  human validation.

## When NOT to use

- **Massive corpora** — hand coding does not scale; use ML with a
  smaller gold-standard test set.
- **Ambiguous constructs** — reliability collapses; refine the
  codebook first.

## Files

- `python/content_analysis_coding.py` — Cohen κ + nominal
  Krippendorff α with missing-data support. Demo (2 raters × 12
  items, 3 categories): **κ = 0.736, α = 0.743** — tentative
  reliability; 3-coder α with two missing values = **0.732**.
- `r/content_analysis_coding.R` — `irr::kappa2` +
  `kripp.alpha`, `irrCAC::gwet.ac1`, `quanteda` (R);
  `krippendorff`, `sklearn.metrics.cohen_kappa_score` (Python).

## Assumptions & caveats

- **Codebook drift** — coder interpretation evolves; retrain
  periodically.
- **Prevalence effects** — rare categories under-represented in
  double-coded subset can hide disagreement (see
  `agreement-beyond-kappa`).
- **Adjudication policy** — pre-register how disagreements are
  resolved (third coder, discussion, senior coder).
- **Report both κ and α** — different weaknesses in different
  settings.

## Related in this repo

- `agreement-beyond-kappa` — modern alternatives to κ (PABAK,
  Gwet AC1).
- `dictionary-methods`, `sentiment-analysis` — automated
  alternatives.

## Run

```
python techniques/content-analysis-coding/python/content_analysis_coding.py
Rscript techniques/content-analysis-coding/r/content_analysis_coding.R
```

**Refs:** Krippendorff, K. *Content Analysis: An Introduction to Its Methodology*, 4th ed., SAGE, 2019; Neuendorf, K.A. *The Content Analysis Guidebook*, 2nd ed., SAGE, 2017.

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
