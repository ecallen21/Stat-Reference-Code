# KWIC / Concordance (Reference §42.20)

McEnery & Hardie (2012), Stefanowitsch & Gries (2003).
**Keyword-in-context (KWIC)** displays every occurrence of a target
word with a configurable window of preceding and following tokens.
The classical corpus-linguistics tool for qualitative pattern
spotting before or after any quantitative analysis.

## When to use

- **Corpus exploration** — inspect how a word is actually used in
  context.
- **Sense disambiguation** — before deciding whether "positive"
  means "positive test result" or "positive attitude".
- **Sanity-check** quantitative keyness / dictionary results.

## When NOT to use

- **Automated pipelines** — KWIC is a human-in-the-loop tool.
- **Very high-frequency words** — the display gets unwieldy;
  sample instead.

## Files

- `python/kwic_concordance.py` — regex-tokenised KWIC with
  configurable window. Demo: 5 "aspirin" mentions in a clinical
  paragraph, each aligned on the keyword with 4-token context.
- `r/kwic_concordance.R` — `quanteda::kwic`,
  `quanteda::textstat_collocations`, `tm` (R);
  `nltk.Text.concordance`, `textacy.extract.kwic`, custom (Python).

## Assumptions & caveats

- **Tokenisation** — sentence-crossing windows may confuse
  qualitative reading; report window in tokens.
- **Case sensitivity** — decide up front; clinical acronyms need
  case-sensitive matches.
- **Regex vs linguistic tokeniser** — a plain regex fails on
  contractions and punctuation; production tools use spaCy /
  quanteda tokenisers.

## Related in this repo

- `collocation-pmi` — statistical collocation tests.
- `keyness-analysis` — quantitative corpus comparison.

## Run

```
python techniques/kwic-concordance/python/kwic_concordance.py
Rscript techniques/kwic-concordance/r/kwic_concordance.R
```

**Refs:** McEnery, T. & Hardie, A. *Corpus Linguistics: Method, Theory and Practice*, Cambridge University Press, 2012; Stefanowitsch, A. & Gries, S.T. "Collostructions: investigating the interaction of words and constructions." *International Journal of Corpus Linguistics*, 2003.

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
