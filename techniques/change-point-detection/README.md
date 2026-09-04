# Change-Point Detection (Reference §38.8)

Chen & Gupta (2012), Killick-Fearnhead-Eckley (2012). Identify abrupt
shifts in distribution parameters — mean, variance, or full
distribution — either retrospectively (offline) or as data arrive
(online).

## Two workhorse offline algorithms

### Binary segmentation (Vostrikova 1981)

Greedy recursion: find the single best split under a CUSUM test, then
recurse on the two halves. `O(n log n)` and simple, but can miss
change points that are close together.

### PELT (Killick-Fearnhead-Eckley 2012)

Exact minimiser of
```
Σ_k [ cost(segment_k) + penalty ]
```
via a pruning rule that removes provably-suboptimal candidate change
points. Expected `O(n)` for a wide class of costs.

## Penalty

BIC-style `penalty = c · σ̂² · log(n)` (typical `c ∈ [1, 3]`);
larger penalty → fewer change points.

## When to use

- **Regime shifts** in economic / climate / biomedical series.
- **Segmenting** signals into piecewise-stationary pieces before
  downstream modelling.
- **Anomaly / structural-break** flags.

## When NOT to use

- **Smooth drift** — a state-space model or spline is more honest.
- **Small samples between change points** — resolution is limited
  by segment length.
- **Online / streaming** with strict latency — use Bayesian OCP
  (Adams-MacKay 2007) or CUSUM/EWMA (see SPC batch).

## Files

- `python/change_point_detection.py` — L2-cost binary segmentation
  and full PELT with pruning. Demo (n=300, three true breaks at
  60, 150, 220): **both algorithms recover 60, 150, 219**.
- `r/change_point_detection.R` — `changepoint`, `ecp`, `bcp` (R);
  `ruptures`, `bayesian_changepoint_detection` (Python).

## Assumptions & caveats

- **Cost mismatch** — using an L2 cost when the true change is in
  variance misses variance-only shifts; use `cpt.meanvar` or a
  Gaussian negative-log-likelihood.
- **Independence** — vanilla algorithms assume independent
  observations; autocorrelated series need pre-whitening or a
  kernel-based cost.
- **Penalty tuning** — under-penalised = spurious change points,
  over-penalised = missed ones. Cross-validation or CROPS
  (Haynes et al. 2017) sweeps a range.
- **Number of change points unknown** — PELT + BIC penalty infers
  it; alternatives include SegNeigh with an explicit `K_max`.

## Related in this repo

- `cusum-charts`, `ewma-charts` — online SPC alternatives.
- `bayesian-changepoint` (if present) — probabilistic online
  detection.
- `state-space-models` — smooth-transition alternative.

## Run

```
python techniques/change-point-detection/python/change_point_detection.py
Rscript techniques/change-point-detection/r/change_point_detection.R
```

**Refs:** Killick, R., Fearnhead, P., & Eckley, I.A. "Optimal detection of change points with a linear computational cost." *JASA*, 2012; Chen, J. & Gupta, A.K. *Parametric Statistical Change Point Analysis*, 2nd ed., Birkhäuser, 2012.

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
