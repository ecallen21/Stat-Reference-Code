# chrF + BERTScore (Reference §25.x extra)

Modern generation-evaluation metrics that address BLEU's known weaknesses
(surface-form only; hostile to paraphrase; token-boundary sensitive).

## chrF (Popović 2015)

Character n-gram F-measure:

```
P = |chrN(cand) ∩ chrN(ref)| / |chrN(cand)|
R = |chrN(cand) ∩ chrN(ref)| / |chrN(ref)|
F_β = (1 + β²) · P · R / (R + β² · P)
```

- **chrF++** adds word n-grams; `β = 2` favours recall.
- Strong for morphologically rich languages where whole words rarely match but character sub-parts do.
- Language-agnostic — no tokeniser needed.

## BERTScore (Zhang et al. 2020)

Cosine-similarity F1 over **contextual** token embeddings from a pretrained
encoder (BERT-large / DeBERTa-xlarge-mnli by default):

```
P = mean_c max_r  cos(vec(c), vec(r))
R = mean_r max_c  cos(vec(c), vec(r))
F1 = 2 · P · R / (P + R)
```

Rewards **semantic similarity**, not just surface overlap. Correlates better
with human judgement than BLEU / ROUGE on most benchmarks.

## Other modern metrics

- **BLEURT** (Sellam 2020) — regression on human ratings, fine-tuned encoder.
- **COMET** (Rei 2020) — encoder + regression trained on WMT DA scores; reference-based (COMET-22) or reference-free (COMET-KIWI); MT SOTA.
- **MAUVE** (Pillutla 2021) — distributional divergence between machine and human generations; open-ended text.
- **LLM-as-judge** (Zheng 2023) — prompt a strong LLM to compare candidates; standard for chatbot evaluation (MT-Bench, AlpacaEval, Arena-Hard).

## When to use

- **Machine translation** — SacreBLEU + chrF + COMET-22 (reference-based) or COMET-KIWI (reference-free).
- **Summarisation** — ROUGE + BERTScore + a targeted human eval.
- **Open-ended chat / creative writing** — MAUVE + LLM-as-judge + humans.
- **Regression testing** — chrF is cheap and deterministic; run in CI.

## Files

- `python/bertscore_chrf_metrics.py` — from-scratch chrF (character n-grams 1–6, β=2) + BERTScore surrogate on toy embeddings. Demo on reference "the cat sat on the mat":
  - **chrF**: exact match 1.000; near-match 0.587; paraphrase 0.310 (surface distance high); unrelated 0.040.
  - **BERTScore surrogate** with toy synonym-close embeddings: exact 1.000; near-match 1.000 (synonymy); **paraphrase 0.993** (recognises "feline rested upon the rug"); unrelated 0.284.
- `r/bertscore_chrf_metrics.R` — `reticulate` + `sacrebleu.corpus_chrf`, `bert_score.score`, `unbabel-comet`; MAUVE (Python).

## Assumptions & caveats

- **BERTScore depends on the encoder** — quality varies with model choice; DeBERTa-xlarge-mnli is the current recommended default.
- **chrF is length-sensitive** — very short strings can be spuriously high; use with a length filter or report side-by-side with token counts.
- **Multi-reference chrF** — use max over references per n-gram to reward covering any one reference.
- **BERTScore inherits the encoder's biases** — a pretrained model that scores "he is a doctor" > "she is a doctor" will treat the two candidates differently.
- **Corpus-level vs sentence-level** — for BLEU / ROUGE, aggregate at the corpus level. For BERTScore, sentence-level is fine.
- **LLM-as-judge caveats** — self-preference bias, position bias, verbosity bias; use pairwise blind comparison + swap positions + calibration.

## Related in this repo

- `bleu-rouge-eval` — the classical baseline this batch complements.
- `text-preprocessing`, `word-embeddings`, `sentence-similarity` — inputs / pipeline.
- `text-generation-decoding`, `transformer-decoder` — the systems whose outputs these metrics evaluate.

## Run

```
python techniques/bertscore-chrf-metrics/python/bertscore_chrf_metrics.py
Rscript techniques/bertscore-chrf-metrics/r/bertscore_chrf_metrics.R
```

**Refs:** Popović, M. "chrF: character n-gram F-score for automatic MT evaluation." *WMT*, 2015; Zhang, T. et al. "BERTScore: evaluating text generation with BERT." *ICLR*, 2020; Rei, R. et al. "COMET: a neural framework for MT evaluation." *EMNLP*, 2020; Zheng, L. et al. "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena." *NeurIPS*, 2023.

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
