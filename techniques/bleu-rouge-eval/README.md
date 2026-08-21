# BLEU + ROUGE (Reference §25.x extra)

Automatic metrics for **text generation** — machine translation (BLEU),
summarisation (ROUGE), captioning, QA. Both compare a candidate string to
one or more reference strings via surface-form overlap.

## BLEU (Papineni et al. 2002)

Modified n-gram precision with a brevity penalty:

```
BLEU-N = BP · exp( Σ_n w_n · log p_n )
p_n    = Σ_g min(count_cand(g), max_ref(g)) / Σ_g count_cand(g)
BP     = 1                     if |c| > |r|
         exp(1 − |r| / |c|)    otherwise
```

- Default is **BLEU-4** with uniform weights (0.25 × 4).
- Multi-reference version takes the max count across references per n-gram.
- Corpus-BLEU aggregates numerators / denominators across sentences (preferred to averaging sentence-BLEU).

## ROUGE (Lin 2004)

Recall-oriented; the ROUGE family:

- **ROUGE-N** — n-gram recall.
- **ROUGE-L** — longest-common-subsequence-based F (recall-weighted, β=1.2 by convention).
- **ROUGE-W** — weighted LCS (favours contiguous matches).
- **ROUGE-S / ROUGE-SU** — skip-bigrams; captures loose word-pair overlap.

## When to use

- **Machine translation benchmarking** — BLEU has been the WMT standard for two decades; SacreBLEU normalises tokenisation.
- **Summarisation evaluation** — ROUGE-1 / ROUGE-2 / ROUGE-L are the classical DUC / TAC / CNN-DailyMail metrics.
- **Regression testing** — cheap, deterministic; catches obvious model regressions.
- **Never as the only metric** — pair with BERTScore / COMET / MAUVE and human evaluation.

## Files

- `python/bleu_rouge_eval.py` — from-scratch BLEU-4 (with brevity penalty and clipping) + ROUGE-L (via LCS DP). Demo (reference "the cat sat on the mat", 5 candidates):
  - Exact match: BLEU 1.000, ROUGE-L 1.000.
  - One-word swap ("a" for "the"): BLEU 0.537, ROUGE-L 0.833 — BLEU harsher.
  - Paraphrase ("a cat is sitting on the mat"): BLEU 0.000 (missing 4-grams), ROUGE-L 0.624.
  - Unrelated ("dog barks loudly"): both 0.
  - Shuffled tokens ("sat cat the mat the on"): BLEU 0.000 (bigrams / trigrams broken), ROUGE-L 0.500.
- `r/bleu_rouge_eval.R` — Python `sacrebleu`, `nltk.translate.bleu_score`, `rouge-score`, `bert_score`, `comet-model` (via `reticulate` from R).

## Assumptions & caveats

- **Surface-form** — synonyms, paraphrases, and grammatical variants penalised harshly. BERTScore and MAUVE partly fix this.
- **Sentence-BLEU is noisy** for short sentences; report corpus-BLEU.
- **Tokenisation matters** — SacreBLEU normalises this; unnormalised BLEU numbers across systems are not comparable.
- **Multiple references** raise scores; single-reference underestimates translation quality.
- **ROUGE-L favours longer candidates** unless a brevity term is added.
- **Not a substitute for human eval** — high BLEU on out-of-domain data may indicate copy-paste rather than translation.

## Related in this repo

- `text-preprocessing` — tokenisation upstream.
- `ngram-language-model` — the n-gram building block.
- `sentence-similarity` — semantic alternative to surface-form BLEU.
- `roc-auc-analysis`, `calibration-scaling` — evaluation-metric siblings in the classification world.

## Run

```
python techniques/bleu-rouge-eval/python/bleu_rouge_eval.py
Rscript techniques/bleu-rouge-eval/r/bleu_rouge_eval.R
```

**Refs:** Papineni, K. et al. "BLEU: a method for automatic evaluation of machine translation." *ACL*, 2002; Lin, C.-Y. "ROUGE: A package for automatic evaluation of summaries." *ACL Workshop on Text Summarization*, 2004; Post, M. "A call for clarity in reporting BLEU scores." *WMT*, 2018 (SacreBLEU).

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
