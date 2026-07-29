# Group-Based Trajectory Modeling (Reference §12.5; also covers §12.6 GMM, §12.7 LCGA)

Assumes the population is a **mixture of K latent trajectory classes**, each with its own polynomial trajectory over time:

```
y_{ij} | (subject i in class k)  ~  N(f_k(t_{ij}), σ²)
f_k(t) = α_{k0} + α_{k1} · t + α_{k2} · t² + ...

π_k = P(subject i in class k)
```

Fit by **EM**:

- **E-step**: posterior `P(class k | subject i's whole trajectory)`.
- **M-step**: update polynomial coefficients per class (weighted OLS on subjects weighted by their posteriors) and update `π_k`.

## GBTM vs. LCGA vs. GMM

| Variant | Ref § | Within-class variation |
|---|---|---|
| **GBTM** | 12.5 | Single shared σ² across classes; no random effects |
| **LCGA** (Latent Class Growth Analysis) | 12.7 | σ² = 0 or fixed; subjects follow their class trajectory exactly (plus noise) |
| **GMM** (Growth Mixture Models) | 12.6 | Random intercepts / slopes WITHIN each class (mixture of LMMs) |

GBTM is the workhorse; LCGA is the strict version; GMM is the flexible version at the cost of more parameters.

## Choosing K

- **BIC** — standard. Smallest BIC wins. `bic_select_k()` fits a grid of K and reports.
- **Entropy** — average posterior certainty. Values > 0.8 typical for clear class separation.
- **Substantive theory** — the classes should correspond to interpretable groups.

## Common problems

- **Class collapse**: with too many K, some classes attract 0 subjects (BIC penalizes this correctly).
- **Local optima**: EM is sensitive to starting values — multiple restarts (`n_restarts` param) mandatory.
- **Overfitting**: BIC penalty helps; also inspect whether classes are meaningfully different.

## Files

- `python/group_based_trajectory.py` — from-scratch EM with `n_restarts` and BIC-based `bic_select_k()` helper. Recovers well-separated latent trajectory classes; collapses redundant classes gracefully.
- `r/group_based_trajectory.R` — pointer to the authoritative `lcmm::hlme()` (unified LCGA/GBTM/GMM) and `crimCV::crimCV()`.

## Assumptions

- Discrete latent-class structure exists. If the true structure is continuous (individual differences on a spectrum), GBTM will over-discretize.
- Enough subjects per candidate class (rule of thumb: ≥ 25 per class).
- Independent subjects.

## Run

```
python techniques/group-based-trajectory/python/group_based_trajectory.py
Rscript techniques/group-based-trajectory/r/group_based_trajectory.R
```

**Refs:** Nagin, D.S. *Group-Based Modeling of Development*, Harvard UP, 2005; Nagin, D.S. & Odgers, C.L. "Group-based trajectory modeling in clinical research." *Ann. Rev. Clin. Psychol.* 6, 109–138, 2010; Muthén, B. & Shedden, K. "Finite mixture modeling with mixture outcomes using the EM algorithm." *Biometrics* 55(2), 463–469, 1999.

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
