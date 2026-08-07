# Credible Intervals, HPD, and ROPE (Reference §14.9, §14.23)

A **credible interval** is a range that contains `(1 − α)` of the posterior probability. Unlike a frequentist confidence interval, the Bayesian version admits the direct probability interpretation ("given the data, there is a 95% probability θ lies here").

## Equal-tail interval (ETI)

```
Pr(θ < θ_L) = Pr(θ > θ_U) = α / 2
```

Simple to compute from posterior draws with the quantile function. Standard default.

## Highest Posterior Density (HPD) / Highest Density Interval (HDI)

The **shortest** interval containing `(1 − α)` posterior mass. For a symmetric posterior HPD = ETI; for a skewed posterior HPD is narrower and always inside the posterior mode. Sensitive to reparameterization — the HPD of `θ` and of `log θ` are not simple transformations of each other.

For **multimodal** posteriors, report the highest-density **region** — a union of intervals that jointly contain the target mass.

## ROPE (Region of Practical Equivalence, Kruschke 2018)

A pre-specified range around a null value considered "practically zero" for the decision at hand — e.g. `|β| < 0.05` on a standardized-effect scale.

```
if 95% HDI is INSIDE ROPE   → accept the null
if 95% HDI is OUTSIDE ROPE  → reject the null
otherwise                    → withhold judgment
```

Directly answers "is the effect big enough to matter?" — the question classical hypothesis testing evades.

## Files

- `python/credible_intervals_hpd.py` — from-scratch ETI, unimodal HPD (shortest-window search on sorted draws), and ROPE decision. Demos: symmetric Normal ETI ≈ HPD (both ~(0.11, 0.69)); skewed lognormal HPD 8.27 wide vs ETI 9.99 wide; three ROPE scenarios yielding reject / withhold / withhold outcomes.
- `r/credible_intervals_hpd.R` — same in base R. Production: `HDInterval::hdi`, `bayestestR::hdi`, `bayestestR::rope`.

## When to use

- **ETI** — default reporting.
- **HPD** — skewed posteriors where the ETI misleads (e.g. contains the mode near one endpoint).
- **ROPE** — decision analysis where "close enough to zero" matters; sequential trials with equivalence stopping rules.

## Caveats

- HPD is not invariant to monotone reparameterization; ETI's endpoints transform naturally.
- The choice of ROPE width is a **substantive** decision, not a statistical one — argue for it from the applied context (minimum clinically important difference, minimum detectable business impact).
- With very informative posteriors any narrow ROPE will lead to rejection; with very diffuse posteriors, judgment is usually withheld.

## Run

```
python techniques/credible-intervals-hpd/python/credible_intervals_hpd.py
Rscript techniques/credible-intervals-hpd/r/credible_intervals_hpd.R
```

**Refs:** Box, G.E.P. & Tiao, G.C. *Bayesian Inference in Statistical Analysis*, Addison-Wesley, 1973 (HPD); Chen, M.-H. & Shao, Q.-M. "Monte Carlo estimation of Bayesian credible and HPD intervals." *J. Comp. Graph. Stat.* 8(1), 69–92, 1999; Kruschke, J.K. "Rejecting or accepting parameter values in Bayesian estimation." *Adv. Meth. Prac. Psych. Sci.* 1(2), 270–280, 2018.

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
