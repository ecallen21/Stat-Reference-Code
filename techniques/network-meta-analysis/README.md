# Network Meta-Analysis (Reference §22.15)

Lumley (2002), Salanti (2012). Simultaneous meta-analysis of
multiple treatments using both **direct** (head-to-head) and
**indirect** (via common comparators) evidence, borrowing strength
across the network.

## Contrast-based random-effects NMA

For study `i` comparing `T_i1` vs `T_i2`:

```
y_i = μ[T_i1] − μ[T_i2] + u_i + ε_i
u_i ~ N(0, τ²),   ε_i ~ N(0, σ_i²)
```

Reference treatment fixed to `μ = 0`; other treatment effects
identified relative to reference. Estimate `(μ, τ)` by REML / ML.

## When to use

- **Multiple treatments** for the same condition — comprehensive
  ranking without head-to-head data for every pair.
- **HTA / NICE decisions** — the standard evidence-synthesis
  framework.

## When NOT to use

- **Disconnected network** — some treatments have no path to the
  reference.
- **Inconsistency** between direct and indirect evidence — check
  first (node-splitting, back-calculation).
- **Very few studies per pair** — sparse networks give unstable
  estimates.

## Files

- `python/network_meta_analysis.py` — RE contrast-based NMA via
  Nelder-Mead on the log-likelihood (custom). Demo (K=3
  treatments, 6 studies per pair, τ=0.10, true μ=(0, 0.5, −0.3)):
  **estimated μ = (0, 0.58, −0.33)**; ranking B > A > C correct.
- `r/network_meta_analysis.R` — `netmeta::netmeta`, `gemtc`,
  `BUGSnet`, `pcnetmeta` (R); custom + `pymare` (Python).

## Assumptions & caveats

- **Transitivity / similarity** — patient populations and
  outcomes across studies are exchangeable.
- **Consistency** between direct and indirect evidence — test via
  node-splitting or design-by-treatment interaction.
- **Reporting** — SUCRA / P-scores for probabilistic ranking;
  network graphs for evidence structure.
- **Sensitivity** — network meta-regression on trial-level
  covariates when heterogeneity is present.

## Related in this repo

- `meta-regression`, `trim-fill` — companion meta-analysis tools.
- `bayesian-glms` — Bayesian NMA underpinning.

## Run

```
python techniques/network-meta-analysis/python/network_meta_analysis.py
Rscript techniques/network-meta-analysis/r/network_meta_analysis.R
```

**Refs:** Lumley, T. "Network meta-analysis for indirect treatment comparisons." *Statistics in Medicine*, 2002; Salanti, G. "Indirect and mixed-treatment comparison, network, or multiple-treatments meta-analysis." *Research Synthesis Methods*, 2012.

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
