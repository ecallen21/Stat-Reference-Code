# Part-of-Speech Tagging (Reference §25.x extra)

Assign each word a syntactic tag (noun / verb / adjective / …) from a fixed
tagset. Standard tagsets:

- **Universal Dependencies** (17 tags) — cross-lingual default.
- **Penn Treebank** (~45 tags) — English fine-grained.
- **Language-specific** — Brown, CLAWS, TIGER, etc.

## Classical HMM tagger

```
y_1, …, y_n  ~ Markov chain over tags        (transitions)
x_i | y_i    ~ Categorical over vocabulary   (emissions)
```

Trained by MLE (with Laplace / Good-Turing smoothing) on a labelled corpus;
decoded with **Viterbi**. Baseline accuracy ~95% on Penn Treebank English.

## Modern taggers

- **BiLSTM-CRF** (Ma & Hovy 2016) — feature-free, works across languages; ~97–98% on Penn Treebank.
- **BERT + linear head** — the SOTA plateau (~97–98%); fine-tuned in minutes.
- **CRF only** — hand-crafted features (suffixes, word-shape, gazetteers); still competitive for morphologically-rich low-resource languages.

## When to use

- **Preprocessing** for parsing, NER, semantic role labelling.
- **Feature engineering** — POS tags as covariates in a classical text classifier.
- **Linguistic analysis** — count nouns / verbs; measure lexical richness.
- **Cross-lingual transfer** — Universal-Dependencies tags transfer across ~200 languages.

## Files

- `python/pos_tagging.py` — from-scratch HMM POS tagger (bigram + Laplace smoothing + Viterbi decoding). Tiny hand-labelled corpus with tags {DT, NN, VB, JJ}. Demo: 100% training-token accuracy; tags 3 held-out sentences correctly, including generalisation to unseen word/tag combinations like "fresh mouse" (JJ NN).
- `r/pos_tagging.R` — `udpipe::udpipe_annotate`, `spacyr::spacy_parse`, `NLP + openNLP` classical Apache OpenNLP.

## Assumptions & caveats

- **Order-1 Markov** in the demo — bigram HMM. Higher-order models (trigram) help but need more data and heavier smoothing.
- **Unknown words** — the demo assigns uniform emission; production HMMs use suffix / word-shape features (`+ing` → VB, `Xxx` → NNP).
- **Ambiguity** — many English words are noun / verb / adjective ambiguous; context resolves it. Viterbi picks the globally best sequence, not the per-token best.
- **Domain shift** — Penn-trained taggers underperform on tweets, clinical notes, legal text; retrain or fine-tune on domain data.
- **Multi-word expressions** — "New York" as one token, "kick the bucket" as one predicate — most taggers don't handle these; use a chunker or MWE-aware tokeniser upstream.
- **Modern SOTA** ~97–98% on English is close to human-agreement ceiling; further gains hard.

## Related in this repo

- `hmm` — the general HMM machinery underlying this tagger.
- `named-entity-recognition` — same sequence-tagging framework, different labels.
- `syntactic-parsing-cky` — the next NLP step after POS tagging.
- `text-preprocessing` — tokenisation input pipeline.

## Run

```
python techniques/pos-tagging/python/pos_tagging.py
Rscript techniques/pos-tagging/r/pos_tagging.R
```

**Refs:** Church, K.W. "A stochastic parts program and noun phrase parser for unrestricted text." *ANLP*, 1988; Brants, T. "TnT: a statistical part-of-speech tagger." *ANLP*, 2000; Ma, X. & Hovy, E. "End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF." *ACL*, 2016.

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
