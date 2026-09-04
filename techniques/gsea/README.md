# Gene Set Enrichment Analysis (Reference §40.4, §40.18)

Subramanian et al. (2005). Given a ranked list of genes (by signed
test statistic or log-fold-change), test whether members of a
predefined **gene set** cluster at the top or bottom of the list.

## Enrichment score

Walk down the ranked list; add `+1/n_hit` for each set member and
subtract `−1/n_miss` for each non-member. The max absolute
cumulative deviation is the **ES**. Signed by walking direction of
the extremum.

Significance via **label permutation** (or sample permutation for
class-based rankings).

## When to use

- **Interpretation** of a DE analysis — group thousands of genes
  into pathways / GO terms.
- **Systems biology** — turn a molecular signature into a hallmark-
  level narrative.

## When NOT to use

- **Very small ranked lists** — permutation nulls are unstable
  below a few hundred genes.
- **No prior gene-set catalogue** in your domain — over-
  representation analysis on ad-hoc lists is a weaker alternative.

## Files

- `python/gsea.py` — signed KS-style enrichment + label-permutation
  p (custom, no library). Demo (200-gene list, 20-gene sets): top-
  biased set **ES = +0.378, p = 0.006**; random set **ES = −0.183,
  p = 0.53** (null-consistent).
- `r/gsea.R` — `fgsea`, `clusterProfiler`, `GSVA`, `topGO`,
  `enrichR` (R); `gseapy`, `goatools`, `gProfiler` API (Python).

## Assumptions & caveats

- **Ranking metric matters** — signed t-stat vs signed FC vs log-
  ratio gives different enrichment profiles.
- **Weighted ES** — the `p` exponent (Subramanian: p=1) weights
  large-magnitude ranks more; extreme p over-emphasises tails.
- **Multiple gene sets** — always BH-correct across sets.
- **Nested / overlapping sets** (GO hierarchy) inflate correlated
  tests; use topology-aware methods (topGO, PADOG).

## Related in this repo

- `differential-expression` — feeds the ranking metric.
- `multiple-testing-corrections` — corrects across many sets.
- `wgcna-coexpression` — module → set → enrichment.

## Run

```
python techniques/gsea/python/gsea.py
Rscript techniques/gsea/r/gsea.R
```

**Refs:** Subramanian, A., Tamayo, P., Mootha, V.K. et al. "Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles." *PNAS*, 2005; Korotkevich, G., Sukhov, V., & Sergushichev, A. "Fast gene set enrichment analysis." *bioRxiv*, 2019 (fgsea).

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
