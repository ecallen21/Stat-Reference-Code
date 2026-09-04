# Keyness Analysis (Reference §42.15)

Dunning (1993), Gabrielatos & Marchi (2012). Test whether words
appear more (or less) often in a **target corpus** than a
**reference corpus**. Standard workhorse: **Dunning's log-
likelihood ratio G²**.

## Statistic

For each word `w`, observed counts `O_T` (target) and `O_R`
(reference):

```
G² = 2 · Σ_c O_c · log(O_c / E_c)
where E_c = (O_T + O_R) · N_c / (N_T + N_R)
```

Chi-square approximation with 1 df; report **log-ratio effect
size** `log(f_T / f_R)` alongside significance.

## When to use

- **Corpus comparison** — target-domain vocabulary discovery
  (clinical vs everyday, party vs party).
- **Feature selection** for downstream classifiers.
- **Corpus linguistics** studies of register / genre.

## When NOT to use

- **Words unique to one corpus** — G² is inflated but not
  informative; report separately.
- **Small samples of either corpus** — chi² approximation is
  poor; use exact Fisher.

## Files

- `python/keyness_analysis.py` — G² + chi²-p + log-ratio effect
  size (custom). Demo (4 clinical target docs vs 5 sports
  reference docs): **"pneumonia" tops G² = 8.21 (p = 0.004)** —
  target-specific vocabulary; "the" flags as reference-only.
- `r/keyness_analysis.R` — `quanteda::textstat_keyness`,
  `quanteda::tokens_keep` + `textstat_frequency` (R);
  `textacy.keyterms`, custom `scipy.stats.chi2` (Python).

## Assumptions & caveats

- **Multiple testing** — BH-correct across the vocabulary; publish
  effect sizes not just significance.
- **Independence of tokens** violated by repeated proper nouns and
  n-grams; consider sentence- or document-level tests.
- **Reference corpus choice** dominates the results — a "general"
  reference is not neutral (BNC vs Google Books changes the story).
- **Stop-word treatment** — either exclude or expect them to
  dominate.

## Related in this repo

- `dictionary-methods`, `collocation-pmi`, `wordfish-scaling` —
  corpus-analysis cousins.
- `chi-square-test` (if present) — the underlying test.

## Run

```
python techniques/keyness-analysis/python/keyness_analysis.py
Rscript techniques/keyness-analysis/r/keyness_analysis.R
```

**Refs:** Dunning, T. "Accurate methods for the statistics of surprise and coincidence." *Computational Linguistics*, 1993; Gabrielatos, C. & Marchi, A. "Keyness: appropriate metrics and practical issues." *CADS International Conference*, 2012.

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
