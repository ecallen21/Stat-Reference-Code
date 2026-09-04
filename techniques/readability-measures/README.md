# Readability Indices (Reference §42.16)

Flesch (1948), Crossley et al. (2008). Formula-based estimates of
the grade level needed to read a passage comfortably. All rely on
syllable / word / sentence counts.

## Common measures

| Index | Formula |
|---|---|
| Flesch Reading Ease | `206.835 − 1.015·ASL − 84.6·ASW` |
| Flesch-Kincaid Grade | `0.39·ASL + 11.8·ASW − 15.59` |
| Gunning Fog | `0.4·(ASL + 100·%complex)` |
| SMOG | `1.043·√(30·polysyllables / n_sent) + 3.129` |
| Coleman-Liau | `0.0588·L − 0.296·S − 15.8` |

`ASL` = avg sentence length in words; `ASW` = avg syllables per
word; complex words = ≥3 syllables; `L`, `S` = per-100-word letter /
sentence counts.

## When to use

- **Health-literacy** assessment of patient-facing materials.
- **Corporate / government** clear-language audits.
- **Cross-document comparison** where absolute grade level is
  less important than relative ranking.

## When NOT to use

- **Deep readability** questions — formula scores ignore cohesion,
  coherence, prior knowledge (Crossley 2008).
- **Non-English** — most indices were fitted on English; use
  language-specific variants.
- **Very short passages** — SMOG requires 30+ sentences; other
  indices are noisy at low `n`.

## Files

- `python/readability_measures.py` — Flesch, Flesch-Kincaid, Fog,
  SMOG, Coleman-Liau with naive syllable count. Demo: simple text
  → **Flesch 110, F-K −1.0, Fog 1.8**; jargon-heavy paragraph →
  **Flesch −121, F-K 36, Fog 38** — indices clearly separate the
  two.
- `r/readability_measures.R` — `quanteda::textstat_readability`,
  `koRpus` (R); `textstat`, `readability` (Python).

## Assumptions & caveats

- **Syllable counting** is a proxy — production tools use
  pronunciation dictionaries (CMU) for accuracy.
- **Sentence detection** — abbreviations trip regex splitters.
- **Grade level ≠ readability** — technical accuracy matters more
  than a low Flesch score for informed consent.
- **Report multiple indices** — no single measure captures
  readability.

## Related in this repo

- `text-preprocessing-pipeline`, `content-analysis-coding` —
  companion analyses.
- `sentiment-analysis` — a different text-property score.

## Run

```
python techniques/readability-measures/python/readability_measures.py
Rscript techniques/readability-measures/r/readability_measures.R
```

**Refs:** Flesch, R. "A new readability yardstick." *Journal of Applied Psychology*, 1948; Crossley, S.A., Greenfield, J., & McNamara, D.S. "Assessing text readability using cognitively based indices." *TESOL Quarterly*, 2008.

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
