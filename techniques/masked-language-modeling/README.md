# Masked Language Modeling — MLM (Reference §25.x extra)

BERT-style pretraining objective (Devlin et al. 2019). Randomly mask ~15% of
tokens and train a **bidirectional** encoder to predict the originals from
context.

## BERT masking recipe

For each token selected for masking:

- **80%** → replaced by `[MASK]`.
- **10%** → replaced by a random token.
- **10%** → left unchanged (still predicted).

The 20% non-`[MASK]` variants mitigate the train-inference mismatch (the
`[MASK]` token never appears at inference).

## Loss

Cross-entropy at masked positions only:

```
L = − (1 / |M|) · Σ_{i ∈ M} log p(x_i | x_{−i})
```

## When to use

- **Encoder pretraining** for downstream classification, tagging, retrieval, embedding.
- **Domain adaptation** — continue MLM on unlabelled target-domain text before fine-tuning.
- **Multilingual pretraining** — XLM-R, mBERT, LaBSE.
- **Not for autoregressive generation** — MLM's bidirectional prediction is different from the causal-LM objective; use `transformer-decoder` / GPT-family for generation.

## Family

| Method | Twist |
|---|---|
| **BERT** (Devlin 2019) | MLM + Next-Sentence-Prediction |
| **RoBERTa** (Liu 2019) | Dynamic masking, drops NSP, longer pretraining |
| **ELECTRA** (Clark 2020) | Replaced-token detection (discriminative); much more sample-efficient |
| **DeBERTa** (He 2020) | Disentangled attention + relative positions |
| **SpanBERT / T5 / BART** | Span masking / span corruption for seq2seq |
| **ModernBERT** (2024) | Long-context, RoPE, mostly-MLM but modernised recipe |

## Files

- `python/masked_language_modeling.py` — minimal MLM: embedding + mean-context bidirectional prediction + linear head (transformer would replace the mean; here we keep it tiny). BERT masking rule with 80/10/10 split. Demo on a 24-sentence toy corpus with 26-word vocabulary: loss drops 3.18 → 0.27 over 200 epochs; mask-filling accuracy 67% at 30% mask rate; correctly fills "the big dog [MASK] quickly" → "runs".
- `r/masked_language_modeling.R` — `torch::nn_module` (custom); `reticulate` + `huggingface transformers` `pipeline('fill-mask')`; `text::textEmbed / textFill`.

## Assumptions & caveats

- **Bidirectional** — MLM cannot directly generate autoregressively; use it for representation, not generation.
- **Mask rate** — Devlin used 15%. Higher rates make the task too hard; lower rates make training inefficient. ModernBERT and others tune this per-corpus.
- **Wordpiece / BPE tokenisation** — real MLM tokenises subwords; a whole word is often 2–4 subword tokens; **whole-word masking** (mask all subwords of a word together) helps.
- **Compute** — pretraining BERT-base takes ~4 days on 16 TPUs; ELECTRA or ModernBERT is more sample-efficient.
- **Domain matching** — a Wikipedia-trained BERT underperforms on clinical / legal / code text; fine-tune with MLM on domain corpus first.
- **Not a language model** — the MLM objective is not `p(x)`; MLM perplexity is not comparable to causal-LM perplexity.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25`, `word-embeddings` — pipeline neighbours.
- `transformer-encoder` — the encoder BERT-style models are trained on.
- `transformer-decoder`, `ngram-language-model` — causal-LM alternatives.
- `sentence-similarity`, `text-classification`, `named-entity-recognition` — downstream tasks that consume MLM-pretrained encoders.
- `contrastive-learning` — modern encoders (E5 / GTE / BGE) combine MLM + contrastive objectives.

## Run

```
python techniques/masked-language-modeling/python/masked_language_modeling.py
Rscript techniques/masked-language-modeling/r/masked_language_modeling.R
```

**Refs:** Devlin, J. et al. "BERT: pre-training of deep bidirectional transformers for language understanding." *NAACL*, 2019; Liu, Y. et al. "RoBERTa: a robustly optimized BERT pretraining approach." *arXiv:1907.11692*, 2019; Clark, K. et al. "ELECTRA: pre-training text encoders as discriminators rather than generators." *ICLR*, 2020.

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
