# Weighted Log-Rank Family + Stratified Log-Rank (Reference §11.4; also covers §11.5, §11.6, §11.7, §11.47, §11.62)

Compares survival distributions between two groups on right-censored data via a weighted rank statistic:

```
U   =  Σ_{event times t_j}  w_j · (d_Aj − E[d_Aj])       under H₀:
E[d_Aj]  =  n_Aj · d_j / n_j
var(d_Aj) = (n_Aj · (n_j − n_Aj) · d_j · (n_j − d_j)) / (n_j² · (n_j − 1))

χ²  =  U² / Σ w_j² · var_j       ~ χ²₁
```

## Weight schemes (which one to use — §11.47, §11.62)

| Scheme | `w_j` | Emphasizes | When |
|---|---|---|---|
| **Log-rank** (Mantel-Cox, §11.4) | 1 | uniform | proportional hazards (most powerful there) |
| **Wilcoxon** (Gehan-Breslow, §11.5) | `n_j` | early events | early differences dominate |
| **Peto-Peto** (§11.5) | `Ŝ_pool(t_j−)` | early events, robust | robust default when unsure |
| **Fleming-Harrington G(ρ, γ)** (§11.6) | `Ŝ^ρ · (1−Ŝ)^γ` | tunable | ρ=1, γ=0 = late; ρ=0, γ=1 = early; ρ=γ=0 = log-rank |
| **Tarone-Ware** (§11.62) | `√n_j` | balance | between log-rank and Wilcoxon |

If hazards cross (obvious non-proportionality), the log-rank test is **under-powered** and misleading — pick a scheme that emphasizes where the difference lies. RMST (see [`rmst`](../rmst)) is often a better default there.

## Stratified log-rank (§11.7)

Sum U and V across strata; single χ² on `(ΣU)² / ΣV`. Use to adjust for a categorical confounder (e.g. age band, site) without modeling it.

## Files

- `python/log_rank_test.py` — unified weighted-log-rank driver over all schemes above + stratified version.
- `r/log_rank_test.R` — from-scratch + `survival::survdiff`.

## Assumptions

- Independent right-censoring within each group.
- **The log-rank ISN'T assumption-free** — it's most powerful when hazards are proportional. Under non-proportional hazards (crossing curves) it loses power dramatically. Always look at the Kaplan-Meier curves before choosing.

## Run

```
python techniques/log-rank-test/python/log_rank_test.py
Rscript techniques/log-rank-test/r/log_rank_test.R
```

**Refs:** Mantel, N. "Evaluation of survival data and two new rank order statistics arising in its consideration." *Cancer Chemother. Rep.* 50(3), 163–170, 1966; Peto, R. & Peto, J. "Asymptotically efficient rank invariant test procedures." *JRSS A* 135(2), 185–207, 1972; Fleming, T.R. & Harrington, D.P. "A class of hypothesis tests for one and two sample censored survival data." *Comm. Stat. A* 10(8), 763–794, 1981; Tarone, R.E. & Ware, J. "On distribution-free tests for equality of survival distributions." *Biometrika* 64(1), 156–160, 1977.

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
