# IBM Model 1 Word Alignment (Reference §25.x extra)

Given a **parallel corpus** (pairs of translated sentences), estimate the
probability that each source word `e_i` is a translation of each target word
`f_j`. Foundational step for statistical machine translation and for
downstream cross-lingual tasks.

## Model

For source sentence `e = (e_1, …, e_l)` and target `f = (f_1, …, f_m)`,
with alignment `a = (a_1, …, a_l)`, `a_i ∈ {0, …, m}` (0 = NULL):

```
P(e, a | f) = (1 / (m + 1)^l) · Π_i t(e_i | f_{a_i})
```

- All alignment positions equally likely (Model 1 assumption; Models 2/HMM/3–5 add positional / fertility structure).
- **NULL token** absorbs source words with no target counterpart.

## EM training

```
E-step: c(e | f) = Σ_pairs Σ_i Σ_j [ t(e_i | f_j) / Σ_k t(e_i | f_k) ] · 1{e_i = e, f_j = f}
M-step: t(e | f) = c(e | f) / Σ_e' c(e' | f)
```

Converges to a unique global maximum from any positive initialisation
(Model 1 is convex; higher models are not).

## Alignment extraction

Given the learned `t`, the max-likelihood alignment is:

```
â_i = argmax_j t(e_i | f_j)
```

## When to use

- **Statistical machine translation** (pre-2015 SOTA; still used as a fallback / feature).
- **Cross-lingual pretraining** — provide alignment supervision.
- **Bitext mining** — filter parallel corpora by mean alignment probability.
- **Bilingual lexicon induction** — pull out `argmax_f t(e | f)` pairs.
- **Cross-lingual NER / annotation projection** — transfer labels via alignment.

## Files

- `python/word_alignment.py` — from-scratch IBM Model 1 EM. Toy English-French corpus with 8 sentence pairs (repeated 10× for stability), 30 EM iterations. Demo:
  - Learned translation probabilities: book→livre 0.999, house→maison 0.999, small→petit(e) 1.0, big→grand(e) 1.0.
  - Alignment of "the small house" ↔ "la petite maison" recovers all three word pairs at probabilities 0.83, 1.00, 1.00.
- `r/word_alignment.R` — no strong native R package; use `reticulate` + Python `fastalign` / `eflomal` / `awesome-align`. Also GIZA++ / Moses (C++).

## Assumptions & caveats

- **Uniform alignment prior** — Model 1 ignores position; Model 2 adds distortion, HMM adds Markov position dependence, Models 3–5 add fertility.
- **1-to-many mapping** — Model 1 aligns each source word to exactly one target word; can't split ("don't" → "do not") or reorder cleanly.
- **NULL is important** — without it, function words (articles, particles) get spurious alignments.
- **EM finds the unique global maximum** for Model 1; higher models have local optima and need Model 1 warm-start.
- **Convergence is slow** on realistic corpora — millions of sentences × 5+ iterations.
- **Modern neural MT** (Transformer encoder-decoder) doesn't need explicit alignment; attention weights serve as soft alignment when needed (e.g. `awesome-align`).

## Related in this repo

- `hmm` — the general HMM machinery; HMM aligner is Model 1 + Markov positions.
- `text-preprocessing`, `tfidf-bm25` — pipeline neighbours.
- `attention-mechanism`, `transformer-encoder` — modern replacement for many alignment-based tasks.

## Run

```
python techniques/word-alignment/python/word_alignment.py
Rscript techniques/word-alignment/r/word_alignment.R
```

**Refs:** Brown, P.F. et al. "The mathematics of statistical machine translation: parameter estimation." *Computational Linguistics* 19(2), 263–311, 1993; Och, F.J. & Ney, H. "A systematic comparison of various statistical alignment models." *Computational Linguistics* 29(1), 19–51, 2003; Dyer, C., Chahuneau, V. & Smith, N.A. "A simple, fast, and effective reparameterization of IBM Model 2." *NAACL*, 2013 (fastalign).

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
