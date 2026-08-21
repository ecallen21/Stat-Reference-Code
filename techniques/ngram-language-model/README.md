# n-gram Language Model (Reference §25.x extra)

Model the joint probability of a sequence as a product of conditional
probabilities on the previous `n − 1` tokens:

```
P(w_1, …, w_T) = Π_t P(w_t | w_{t−n+1}, …, w_{t−1})
```

## Smoothing

Raw MLE assigns zero probability to any unseen n-gram, giving infinite
perplexity on held-out data. Standard smoothers:

- **Laplace (add-α)**: `(count(h,w) + α) / (count(h) + α · V)`. Simple; over-smooths.
- **Good-Turing**: reassign mass from n-grams seen `r+1` times to those seen `r` times.
- **Witten-Bell**: back-off + interpolation with lower-order context.
- **Kneser-Ney** (best pre-neural): interpolated with a **continuation probability** based on how many distinct contexts a word appears in — not just its unigram count. The default smoother for classical LMs.

## Perplexity

Standard held-out evaluation metric:

```
PPL = exp( − (1 / N) · Σ log P(w_i | h_i) )
```

Lower = better. Perplexity 100 ~ average of picking among 100 words at each step.

## When to use

- **Baseline for text generation / autocomplete** — n-gram LMs are microsecond-per-token, no GPU.
- **Statistical machine translation** (pre-2015 SOTA).
- **Fluency scoring** — reject grammatically-implausible outputs.
- **Rare-language ASR / OCR** — where transformer LMs aren't pretrained.
- **Baseline for LLM perplexity comparisons** — kenlm 5-gram is the standard reference.

## Files

- `python/ngram_language_model.py` — from-scratch unigram / bigram / trigram with Laplace + simplified interpolated Kneser-Ney bigram. Perplexity demo:
  - 1-gram Laplace = 12.25, 2-gram = 4.14, 3-gram = 4.97 (data-starved: trigram sparsity outweighs context gain here).
  - Kneser-Ney bigram = 2.23 (much sharper).
  - Sampled trigram sentences plausible for the small corpus.
- `r/ngram_language_model.R` — `quanteda::tokens_ngrams`, `text2vec::create_dtm(ngram=…)`; Python `nltk.lm.MLE / KneserNeyInterpolated`, `kenlm` for production KN.

## Assumptions & caveats

- **Markov assumption** — only the previous `n − 1` tokens matter. Fails for long-range coherence.
- **Sparsity explodes exponentially with n** — trigrams over 20k vocabulary have 8×10¹² possible tuples.
- **Smoothing choice dominates** — Kneser-Ney beats Laplace by 5–30× on realistic corpora.
- **Vocabulary closure** — always include `<UNK>` and pad with `<s>` / `</s>`; subword tokenisation reduces OOV further.
- **Perplexity depends on tokenisation** — word-level vs BPE vs character-level PPL numbers aren't comparable.
- **Transformer causal LMs** (GPT / LLaMA family) beat n-gram PPL by 10–30× on general text at 10⁴–10⁶× the compute; n-gram LMs remain useful when latency / memory / no-GPU is the constraint.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25` — pipeline neighbours.
- `hmm`, `pos-tagging`, `named-entity-recognition` — statistical sequence models built on the same counting-with-smoothing idea.
- `topic-modeling-lda`, `topic-coherence-eval` — perplexity as an evaluation companion.
- `transformer-encoder` — the modern replacement.

## Run

```
python techniques/ngram-language-model/python/ngram_language_model.py
Rscript techniques/ngram-language-model/r/ngram_language_model.R
```

**Refs:** Chen, S.F. & Goodman, J. "An empirical study of smoothing techniques for language modeling." Harvard TR-10-98, 1998; Kneser, R. & Ney, H. "Improved backing-off for m-gram language modeling." *ICASSP*, 1995; Jurafsky, D. & Martin, J.H. *Speech and Language Processing*, 3rd ed. draft, Chs 3.

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
