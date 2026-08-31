# Non-negative Matrix Factorisation (Reference §25.2)

Lee & Seung (1999) — factorise a non-negative matrix `V (n × d)` as
`V ≈ W H` with `W (n × k)` and `H (k × d)` both non-negative.
Non-negativity forces **additive combinations** and yields interpretable
"parts-based" representations: topics from documents, spectra from
mixed samples, gene programmes from expression, facial parts from
images.

## Multiplicative updates (Lee-Seung 2001)

```
H ← H · (Wᵀ V) / (Wᵀ W H + ε)
W ← W · (V Hᵀ) / (W H Hᵀ + ε)
```

Guaranteed non-increasing Frobenius objective; maintains
non-negativity by construction.

## When to use

- **Topic modelling** — bag-of-words matrix with `k` topics.
- **Hyperspectral / mass-spec unmixing** — non-negative component
  spectra.
- **Gene-expression** — additive expression programmes.
- **Image parts** — facial features (Lee-Seung's original demo).

## When NOT to use

- **Signed / centred data** — non-negativity is a hard constraint.
- **Very sparse `V` with strong zeros** — probabilistic LDA / PLSA
  may fit better.
- **Uniqueness sensitive applications** — NMF factors are not unique;
  results depend on initialisation.

## Files

- `python/nmf.py` — from-scratch multiplicative-update NMF with
  Frobenius loss. Demo on a 6-doc × 5-word synthetic topic matrix
  (`animals` and `law`): **topic 0 = {law, lawyer, trial}, topic 1 =
  {dog, cat, lawyer}**; reconstruction error 98.4 → 10.2 after 300
  iterations.
- `r/nmf.R` — R `NMF` package (Gaujoux), `RcppML`, `MOFA`;
  `sklearn.decomposition.NMF`, `nimfa`, `tensorly` (Python).

## Assumptions & caveats

- **Non-negative data only** — shift or split into positive/negative
  parts first.
- **Rank `k`** — determined by AIC / stability / interpretability;
  no closed-form choice.
- **Initialisation matters** — random / NNDSVD / warm-start; multiple
  restarts and pick best objective.
- **Sparsity** — plain NMF does not enforce sparse `W` or `H`; use
  sparse NMF (Hoyer 2004) if needed.
- **Loss choice** — Frobenius is default; KL divergence recovers PLSA;
  Itakura-Saito for audio.
- **Not identifiable** — factors defined up to scale × permutation.

## Related in this repo

- `ica` — independence-based factorisation (signed).
- `sparse-pca`, `dictionary-learning`, `variational-autoencoder` —
  other latent-factor families.
- `topic-modeling-lda` (if present) — probabilistic sibling.
- `latent-class-analysis` — discrete latent alternative.
- `correspondence-analysis` — chi-square factorisation for contingency.

## Run

```
python techniques/nmf/python/nmf.py
Rscript techniques/nmf/r/nmf.R
```

**Refs:** Lee, D.D. & Seung, H.S. "Learning the parts of objects by non-negative matrix factorization." *Nature*, 1999; Lee, D.D. & Seung, H.S. "Algorithms for non-negative matrix factorization." *NeurIPS*, 2001.

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
