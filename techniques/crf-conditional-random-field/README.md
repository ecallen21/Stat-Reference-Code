# Conditional Random Field — CRF (Reference §47.157)

Lafferty, McCallum & Pereira (2001). Linear-chain CRF models the
**conditional** distribution:

    p(y | x) ∝ exp( Σ_t [ f(x_t, y_t) + g(y_{t−1}, y_t) ] ).

Unlike HMMs (which model p(x, y)), CRFs are **discriminative** and
allow arbitrary overlapping features of the entire input x. Train
by maximising conditional log-likelihood (gradient uses
forward-backward posteriors); predict via Viterbi.

## Files

- `python/crf_conditional_random_field.py` — from-scratch linear-
  chain CRF with hand-derived gradients via forward-backward. Toy
  chunking task (3 tags, 4 word-shape features, 200 sequences of
  length 20, transitions favour B → I → O):
  - **CRF Viterbi token accuracy = 0.670**
  - Per-token argmax (no transitions) = 0.625 — CRF picks up 4.5
    pt from the transition structure.
- `r/crf_conditional_random_field.R` — recommends `crfsuite`
  (CRFsuite R wrapper).

## When to use

- **Sequence labelling**: NER, POS tagging, chunking, gene
  finding.
- **Discriminative training** where features are cheap to compute
  and generative modelling is overkill.
- **Structured prediction** where output labels have known
  dependencies (adjacent labels, tree structure via TreeCRF).

## When NOT to use

- **When neural sequence models are affordable** — BiLSTM-CRF /
  Transformer-CRF beat plain CRFs on most benchmarks.
- **Very rich, high-cardinality feature spaces** — L1/L2 penalty
  needed; even then O(T · K²) per step is costly for large K.
- **Non-sequential dependencies** — use general MRFs (loopy BP or
  Gibbs).

## Assumptions & caveats

- **First-order dependency** (y_t | y_{t−1}) is standard; higher-
  order dramatically inflates K.
- **Log-space arithmetic** required; forward-backward is O(T · K²).
- **L1 / L2** regularisation prevents feature explosion.
- **Global normalisation** avoids the label-bias problem of MEMMs.

## Related in this repo

- `viterbi-algorithm` — the decoder used at test time.
- `hmm`, `baum-welch-hmm` — generative cousin.
- `maximum-entropy` — logistic-regression sibling for
  unstructured classification.
- `information-bottleneck`, `energy-based-models` — related
  energy / log-linear formulations.

## Run

```
python techniques/crf-conditional-random-field/python/crf_conditional_random_field.py
Rscript techniques/crf-conditional-random-field/r/crf_conditional_random_field.R
```

**Refs:** Lafferty, J., McCallum, A. & Pereira, F. C. "Conditional random fields: Probabilistic models for segmenting and labeling sequence data." *ICML*, 2001; Sutton, C. & McCallum, A. "An introduction to conditional random fields." *Foundations and Trends in ML* 4(4), 2012; Sarawagi, S. & Cohen, W. W. "Semi-Markov CRFs for information extraction." *NIPS*, 2005.

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
