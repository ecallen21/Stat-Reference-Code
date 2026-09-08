# LambdaMART Learning-to-Rank (Reference §47.129)

Burges (2010). Combines RankNet + LambdaRank + boosted trees:

- **RankNet** (Burges 2005): pairwise cross-entropy loss on score
  differences.
- **LambdaRank**: scale each pairwise gradient by |ΔNDCG_{ij}|.
- **LambdaMART**: gradient-boosted trees on the aggregated λ.

The listwise gradient at document i in query q:

    λᵢ = Σ_{j: rel(i)≠rel(j)} sign(rel(i)−rel(j)) · σ(−|sᵢ−sⱼ|) · |ΔNDCG_{ij}|.

## Files

- `python/learning_to_rank_lambdamart.py` — LambdaMART-lite:
  linear boosters fit on λ gradients per query with per-swap NDCG
  weighting. Demo (8 queries × 10 docs × 5 features, 50 rounds):
  - **Mean NDCG@10 (LambdaMART) = 0.996**
  - Random ordering baseline    = 0.691.
- `r/learning_to_rank_lambdamart.R` — `xgboost::xgb.train(objective
  = "rank:pairwise" or "rank:ndcg")`, `lightgbm::lgb.train(rank_xendcg)`
  (R + Python); `TensorFlow Ranking` (Python).

## When to use

- **Search / recommendation ranking** with graded relevance.
- **Multi-query training** where per-query metrics matter.
- **When NDCG / MAP / MRR is the online metric**.

## When NOT to use

- **Binary relevance without ranking signal** — logistic
  regression suffices.
- **Very small #queries** — high-variance NDCG gradients.
- **Point-wise regression targets** — regression / classification
  trees are simpler.

## Assumptions & caveats

- **Grouped train data** — must pass query IDs so pairs stay within
  queries.
- **Loss stability**: |ΔNDCG| can be zero if two docs share the same
  relevance in the same rank — LambdaRank downweights ties.
- **Hyperparameters**: sigma (steepness), max_depth of trees,
  learning rate, num_trees.
- **Position bias** — combine with unbiased-LTR (Ai 2018) for
  click-log data.

## Related in this repo

- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `gradient-boosting`, `catboost-ordered-boosting`,
  `random-forest`, `bart-bayesian-additive-regression-trees` —
  gradient-boosting family.
- `plackett-luce-ranking`, `bradley-terry`, `elo-glicko-rating` —
  probabilistic ranking cousins.
- `discrimination-calibration`, `f1-optimal-threshold`,
  `youden-optimal-cutpoint` — evaluation-metric neighbours.
- `matrix-factorization-als-recsys`,
  `neural-collaborative-filtering`, `reciprocal-rank-fusion` —
  recsys / retrieval cousins.

## Run

```
python techniques/learning-to-rank-lambdamart/python/learning_to_rank_lambdamart.py
Rscript techniques/learning-to-rank-lambdamart/r/learning_to_rank_lambdamart.R
```

**Refs:** Burges, C.J.C. "From RankNet to LambdaRank to LambdaMART: An overview." *Microsoft Research Technical Report MSR-TR-2010-82*, 2010.

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
