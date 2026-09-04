# Clinical NLP (Reference §42.5)

Savova et al. (2010), Chapman et al. (2001), Alsentzer et al. (2019).
Clinical text mining extracts **concepts** (drugs, diagnoses,
procedures, labs) plus their **modifiers** — negation, uncertainty,
family history, historical — from free-text notes.

## The NegEx algorithm

Chapman 2001: for each candidate concept, scan a window preceding
(and optionally following) the mention for **negation triggers**
(`no`, `denies`, `without`, `ruled out`, ...) or **uncertainty
triggers** (`possible`, `probable`, ...). Flag the concept as
`NEGATED`, `UNCERTAIN`, or `POSITIVE`. Restrict the window to the
current sentence to avoid cross-sentence bleed.

## When to use

- **EHR chart-review automation** — cohort identification, adverse-
  event surveillance, phenotype algorithm development.
- **Clinical-trial screening** — quickly find eligible patients from
  free-text notes.

## When NOT to use

- **Structured EHR data alone** covers your question — no NLP
  needed.
- **Highly specialised subdomain** without a fitted lexicon or
  transformer — off-the-shelf tools may miss most concepts.

## Files

- `python/clinical_nlp.py` — small clinical lexicon + NegEx-style
  scanning restricted to the current sentence. Demo (4 notes): all
  status labels correct — "cough / fever / aspirin" POSITIVE;
  "No evidence of pneumonia" NEGATED; "Possible asthma" UNCERTAIN;
  "Denies chest pain" NEGATED; "History of metformin" POSITIVE.
- `r/clinical_nlp.R` — `clinspacy`, `spacyr`, `tidytext` (R);
  `scispacy`, `medspacy`, `negspacy`, `transformers` (BioBERT /
  ClinicalBERT) (Python); external: Apache cTAKES.

## Assumptions & caveats

- **Lexicon coverage** dominates results — UMLS linking
  (`scispacy.linker`) is the standard for real projects.
- **Sentence boundary** — this reference uses `. ? !`; abbreviations
  like "Mr." or "e.g." need a proper sentence splitter.
- **Beyond negation** — ConText (Harkema 2009) also flags
  historical, hypothetical, family-history modifiers.
- **Evaluation** — always report PPV / sensitivity / specificity vs
  a chart-review gold standard.

## Related in this repo

- `named-entity-recognition`, `relation-extraction` — the general
  NER pipeline.
- `text-preprocessing-pipeline` — tokenisation upstream.
- `agreement-beyond-kappa` — inter-annotator reliability on labels.

## Run

```
python techniques/clinical-nlp/python/clinical_nlp.py
Rscript techniques/clinical-nlp/r/clinical_nlp.R
```

**Refs:** Savova, G.K., Masanz, J.J., Ogren, P.V. et al. "Mayo clinical text analysis and knowledge extraction system (cTAKES): architecture, component evaluation and applications." *JAMIA*, 2010; Chapman, W.W., Bridewell, W., Hanbury, P., Cooper, G.F., & Buchanan, B.G. "A simple algorithm for identifying negated findings and diseases in discharge summaries." *Journal of Biomedical Informatics*, 2001; Alsentzer, E., Murphy, J.R., Boag, W. et al. "Publicly available clinical BERT embeddings." *Clinical NLP Workshop*, 2019.

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
