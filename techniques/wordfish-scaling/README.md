# Wordfish Text Scaling (Reference §42.18)

Slapin & Proksch (2008). **Unsupervised** positioning of documents
on a single latent dimension using word counts. Compared to
**Wordscores** (Laver-Benoit-Garry 2003), Wordfish does **not**
require pre-labelled reference texts.

## Model

```
y_ij ~ Poisson(exp(α_i + ψ_j + β_j · ω_i))
```

- `α_i` — document verbosity offset.
- `ω_i` — document position on the latent scale (identifier: mean 0,
  sd 1).
- `ψ_j` — word baseline frequency.
- `β_j` — word discrimination on the latent scale (positive =
  right-leaning, negative = left-leaning, or whatever the poles
  represent).

Estimated by **alternating Poisson regressions** on documents and
words, with an identification constraint on `ω`.

## When to use

- **Political-text scaling** — party manifestos, parliamentary
  speeches on ideological dimensions.
- **Any longitudinal position estimation** where the underlying
  scale is one-dimensional and words drift.
- **Unsupervised alternative** to Wordscores when no gold-standard
  reference documents exist.

## When NOT to use

- **Multiple latent dimensions** — Wordfish is 1-D by construction;
  use STM or embedding-based methods for multi-dimensional scaling.
- **Very small corpora** — Poisson estimation is noisy; interpret
  positions with care.

## Files

- `python/wordfish_scaling.py` — Poisson alternating-Newton
  estimator for `(α, ω, ψ, β)` with `ω` identified to mean 0,
  sd 1. Demo (5 documents on labour vs market rhetoric): union /
  labour docs get **ω = −0.59, −1.40**; market / freedom docs get
  **+1.55, +0.53**; centrist doc **−0.10**. Top-β words
  (freedom / liberty / enterprise) on the right; bottom-β words
  (rights / equality / reform) on the left.
- `r/wordfish_scaling.R` — `quanteda.textmodels::textmodel_wordfish`
  + `textmodel_wordscores` (R); custom Python.

## Assumptions & caveats

- **Single latent dimension** — assumes one underlying axis
  explains word choice.
- **Poisson assumption** — extra-Poisson variability requires
  negative-binomial extensions.
- **Identification** — the direction of `ω` is arbitrary; anchor
  it with two "polar" documents post-fit.
- **Rare words** dominate `β` in small corpora; pre-trim or use
  regularisation.
- **Static positions** — extensions (dynamic Wordfish) let `ω`
  evolve over time.

## Related in this repo

- `structural-topic-model` — covariate-aware topic prevalence.
- `dictionary-methods`, `keyness-analysis`, `sentiment-analysis`
  — competing text-scoring approaches.

## Run

```
python techniques/wordfish-scaling/python/wordfish_scaling.py
Rscript techniques/wordfish-scaling/r/wordfish_scaling.R
```

**Refs:** Slapin, J.B. & Proksch, S.-O. "A scaling model for estimating time-series party positions from texts." *American Journal of Political Science*, 2008; Laver, M., Benoit, K., & Garry, J. "Extracting policy positions from political texts using words as data." *APSR*, 2003.

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
