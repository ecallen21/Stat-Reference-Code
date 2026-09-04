# Information Bottleneck (Reference §34.12)

Tishby, Pereira & Bialek (1999). Given input `X` and target `Y`, learn
a compressed representation `T` that **maximises `I(T; Y)`** while
**keeping `I(T; X)` small**:

```
L_IB(T)  =  I(T; X)  −  β · I(T; Y).
```

`β` trades **compression** for **prediction**. `β = 0` gives a
trivial compressor; `β → ∞` retains all information about `X` used to
predict `Y`.

## Iterative Blahut-Arimoto updates

```
p(t | x) ← p(t)/Z · exp( −β · KL( p(y | x) ‖ p(y | t) ) )
p(t)      ← Σ_x p(x) p(t | x)
p(y | t)  ← Σ_x p(y | x) p(t | x) p(x) / p(t)
```

Related: **Deep IB / VIB** (Alemi 2017) uses variational bounds for
neural nets; **IB theory of deep learning** (Tishby-Zaslavsky 2015).

## When to use

- **Interpretable representation learning** with a task target.
- **Feature compression** for downstream classification / regression.
- **Understanding what a neural layer keeps about `Y`** — the
  info-plane analysis.

## When NOT to use

- **Unlabelled `Y`** — needs a target signal.
- **Continuous high-dim** without a variational bound — the exact IB
  is intractable.
- **You just need PCA / autoencoder** — IB is heavier machinery.

## Files

- `python/information_bottleneck.py` — discrete IB via Blahut-Arimoto
  on synthetic (X, Y) with 3-cluster joint. Sweep `β`:
  - `β=0.1-1.0`: trivial compressor (both MIs ≈ 0).
  - `β=3.0`: `I(T; X) = 1.09`, `I(T; Y) = 0.73`.
  - `β=10.0`: `I(T; X) = 1.10`, `I(T; Y) = 0.74` (near full
    `I(X; Y) = 0.75`).
- `r/information_bottleneck.R` — `reticulate` + Python
  `information-bottleneck` / `deep-info-bottleneck`.

## Assumptions & caveats

- **Bottleneck size `T`** — larger `T` allows more compression /
  prediction resolution.
- **Non-convex objective** — random restart helps; iterations can
  stall in trivial local optima at small `β`.
- **Continuous IB** requires variational bounds (VIB) or kernel-based
  MI estimators.
- **DPI / dual bounds** — Poole 2019 shows caveats of estimating MI
  in high-dim with samples.

## Related in this repo

- `mutual-information`, `conditional-mutual-info`,
  `transfer-entropy`, `kl-divergence` — sibling info-theoretic
  quantities.
- `contrastive-learning`, `variational-autoencoder`,
  `autoencoder` — deep-learning cousins.
- `feature-store`, `principal-component-regression` — task-agnostic
  compression alternatives.

## Run

```
python techniques/information-bottleneck/python/information_bottleneck.py
Rscript techniques/information-bottleneck/r/information_bottleneck.R
```

**Refs:** Tishby, N., Pereira, F.C. & Bialek, W. "The information bottleneck method." *37th Allerton Conf on Communication*, 1999; Alemi, A. et al. "Deep variational information bottleneck." *ICLR*, 2017; Tishby, N. & Zaslavsky, N. "Deep learning and the information bottleneck principle." *IEEE ITW*, 2015.

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
