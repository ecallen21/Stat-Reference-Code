# Ranked-Set Sampling (RSS) (Reference §27.9)

McIntyre (1952). When precise measurement is expensive but a cheap
covariate can **rank** a small set of units:

1. Draw `m` units at random; rank them cheaply (visual triage,
   proxy variable).
2. Measure the **smallest** accurately.
3. Repeat with fresh sets; measure the 2nd smallest, then 3rd,
   ..., then `m`-th → one "cycle" of `m` measurements.
4. Do `r` cycles for `n = m · r` accurate measurements.

Estimator: sample mean. Takahasi-Wakimoto (1968) show that under
**perfect ranking**

    Var(ȳ_RSS)  ≤  Var(ȳ_SRS) / m

so a 4-fold RSS can quadruple efficiency for the same measurement
budget.

## Files

- `python/ranked_set_sampling.py` — from-scratch RSS + SRS sample
  means over 500 Monte-Carlo replicates, varying `m` and ranking
  noise. Demo (N=20 000 population, m ∈ {2, 4, 8}, r=40 cycles):
  perfect-ranking variance reductions 1.31×, 2.40×, 3.44×; noisy
  ranking (σ_noise=1.5) still gives 1.13×, 1.57×, 1.57×.
- `r/ranked_set_sampling.R` — `RSSampling`, `rssampling`,
  `sampling` (R); from-scratch (Python).

## When to use

- **Expensive precise measurement, cheap ranking** — soil sampling,
  forestry (tree height), water-quality assays, chart-review-then-
  interview studies.
- **Environmental / ecological surveys** — rare taxa where you can
  eye which sample deserves precise identification.
- **Budgeted survey design** — fixed lab-analysis budget; RSS
  extracts more info than SRS.

## When NOT to use

- **Ranking is noisier than measurement** — RSS gains vanish and
  can become losses.
- **Small population** — set-of-m needs 2m² independent draws per
  cycle; hard with a tiny frame.
- **You want distribution shape, not just mean** — RSS estimators
  for higher moments have less-transparent gains; check.

## Assumptions & caveats

- **Independent sets** — the m² units drawn per cycle should be
  distinct and independent.
- **Judgment ranking** — accuracy of the cheap rank drives the
  variance gain; poor ranking wastes effort.
- **Median RSS / balanced RSS** — variants for skewed populations
  and mean-vs-median estimands.
- **Missing / broken ranks** — handle via truncation or median RSS.

## Related in this repo

- `two-stage-cluster-sampling`, `complex-survey-design` — design-
  cousins.
- `stratified` sampling and post-stratification — variance-reduction
  siblings.
- `mde-sample-size`, `cuped-variance-reduction` — variance-reduction
  cousins in experiments.

## Run

```
python techniques/ranked-set-sampling/python/ranked_set_sampling.py
Rscript techniques/ranked-set-sampling/r/ranked_set_sampling.R
```

**Refs:** McIntyre, G.A. "A method of unbiased selective sampling using ranked sets." *Australian Journal of Agricultural Research*, 3(4): 385-390, 1952; Takahasi, K. & Wakimoto, K. "On unbiased estimates of the population mean based on the sample stratified by means of ordering." *Ann. Inst. Stat. Math.*, 20: 1-31, 1968; Chen, Z., Bai, Z. & Sinha, B.K. *Ranked Set Sampling: Theory and Applications*, Springer, 2004.

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
