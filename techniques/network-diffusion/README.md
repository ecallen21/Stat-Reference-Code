# Network Diffusion Models (Reference §24.8)

Discrete-time simulations of how something (a virus, a behaviour, information)
spreads across a network.

## Compartmental epidemic models

| Model | States | Rule |
|---|---|---|
| **SI** | S → I | Each S neighbour of an I becomes I with probability `β` per step |
| **SIS** | S → I → S | Add `γ` recovery back to S |
| **SIR** | S → I → R | Recovery to permanent R |
| **SEIR** | S → E → I → R | Add exposed-but-not-infectious latent state |

Basic reproduction number `R₀ ≈ β · <k> / γ` on a well-mixed graph;
network structure raises the threshold above the mean-field limit.

## Diffusion of behaviour / information

| Model | Rule |
|---|---|
| **Independent cascade** | Each newly-active node tries **once** to activate each neighbour with per-edge probability `p_uv` |
| **Linear threshold** | Node activates when Σ (active-neighbour weights) ≥ threshold θ_i |
| **Bass model** | mean-field mix of innovation `p` and imitation `q` (aggregate, not node-level) |

**Cascade** models point-to-point contagion (disease-like), **threshold** models complex social contagion where multiple reinforcing sources are needed.

## When to use

- **Epidemic forecasting** — SIR / SEIR calibrated to observed cases.
- **Marketing / adoption** — IC / LT for viral campaigns; select seeds to maximise reach.
- **Rumour / misinformation** — mixed IC + decay.
- **Influence maximisation** — pick `k` seeds that maximise expected active set (Kempe-Kleinberg 2003; greedy on submodular expected spread).

## Files

- `python/network_diffusion.py` — from-scratch SI / SIR / IC / LT simulators. Demo (ER graph n=100, mean degree ~5, seeds = {0, 1, 2}): SI(β=0.15) reaches 100/100 in 23 steps; SIR(β=0.15, γ=0.10) peaks at 51 infected, final 93 recovered; IC(p=0.10) 4/100 (single-shot stochastic); LT (θ ~ U(0.1, 0.5)) reaches 84/100 (deterministic tipping cascade).
- `r/network_diffusion.R` — `EpiModel::netsim`, `netdiffuseR::rdiffnet`.

## Assumptions & caveats

- **Discrete time**; continuous-time (Gillespie) simulation is more realistic for epidemics but heavier.
- **Homogeneous rates** here; real applications need per-edge / per-node parameters (age, contact intensity).
- **No behavioural response** — an epidemic model with fixed `β` misses interventions, awareness, distancing. Add time-varying `β(t)` for realism.
- **Stochastic** — always average over many runs; single-run outcomes vary widely (compare IC 4 vs LT 84 on the same seeds and graph above).
- **Complex vs simple contagion** — LT thresholds > 0.5 spread very differently from IC; pick the model that matches the substance.

## Run

```
python techniques/network-diffusion/python/network_diffusion.py
Rscript techniques/network-diffusion/r/network_diffusion.R
```

**Refs:** Kermack, W.O. & McKendrick, A.G. "A contribution to the mathematical theory of epidemics." *Proc. R. Soc. A* 115, 700–721, 1927; Granovetter, M. "Threshold models of collective behavior." *Amer. J. Sociol.* 83(6), 1420–1443, 1978; Kempe, D., Kleinberg, J. & Tardos, É. "Maximizing the spread of influence through a social network." *KDD*, 2003.

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
