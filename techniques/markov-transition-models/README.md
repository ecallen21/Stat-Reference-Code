# Markov Transition Models for Longitudinal Categorical Data (Reference §12.9)

For a **discrete-state** variable observed repeatedly per subject (health status, employment, disease stage), a first-order Markov model assumes the next state depends **only on the current state**:

```
P(Y_{t+1} = j | Y_t = i, entire history)  =  P(Y_{t+1} = j | Y_t = i)  =  P_ij
```

## MLE of the transition matrix

Pool all observed transitions across subjects:

```
P̂_ij  =  n_ij / n_i.
```

where `n_ij` = number of `i → j` transitions in the pooled data. Extremely simple, no optimization required.

## Stationary distribution

Long-run proportion of time spent in each state:

```
π P  =  π                (π is the left eigenvector for eigenvalue 1)
```

## Order test (§12.9)

LR test: are transitions really first-order Markov, or just marginal (zero-order)?

```
LR  =  2 · (ll_first_order − ll_zero_order)   ~   χ²_{(K − 1)²}
```

Small p ⇒ the previous state matters — first-order structure exists.

## Extensions (not implemented; add as needed)

- **Non-homogeneous transitions**: `P_ij(t)` varies with time — fit a separate P per period.
- **Covariate-dependent transitions**: fit a multinomial GLM per row of P (destination as outcome, covariates as predictors).
- **Higher-order Markov**: `P(Y_{t+1} | Y_t, Y_{t-1})` — extend the state space to pairs.

## Files

- `python/markov_transition_models.py` — MLE transition matrix + stationary distribution + first-order-vs-zero-order LR test. Recovers the true P from a simulated 3-state chain within ~1% per cell on n=300 subjects × 6 timepoints.
- `r/markov_transition_models.R` — from-scratch base R + optional `markovchain::markovchainFit`.

## Assumptions

- **Markov property** (memoryless given current state). If sequences show long-range dependence, use hidden-Markov or higher-order models.
- **Homogeneous** transitions across time (unless you fit period-specific matrices).
- Independent subjects.

## Run

```
python techniques/markov-transition-models/python/markov_transition_models.py
Rscript techniques/markov-transition-models/r/markov_transition_models.R
```

**Refs:** Kalbfleisch, J.D. & Lawless, J.F. "The analysis of panel data under a Markov assumption." *JASA* 80(392), 863–871, 1985; Diggle, P., Heagerty, P., Liang, K.-Y. & Zeger, S. *Analysis of Longitudinal Data*, 2nd ed., Oxford, 2002 (Ch. 10); Bishop, Y.M.M., Fienberg, S.E. & Holland, P.W. *Discrete Multivariate Analysis*, MIT Press, 1975 (Ch. 7).

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
