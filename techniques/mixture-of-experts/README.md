# Mixture of Experts — MoE (Reference §27.x extra)

Sparsely-Gated MoE (Shazeer 2017) / Switch Transformer (Fedus 2022). Route
each input token to a small subset of specialised **expert** networks, keeping
per-token compute low while total parameter count can be enormous.

## Sparse top-k gating

```
gate(x)   = softmax(W_g x)                     ← E logits over experts
top-k(x)  = keep the k highest logits, renorm  ← activate only k experts
output    = Σ_{e ∈ top_k} gate_e(x) · Expert_e(x)
```

- **k = 1** — Switch Transformer (Fedus 2022).
- **k = 2** — Mixtral-8×7B, DeepSeek-V2, GShard.
- **capacity factor** — cap per-expert token budget to keep TPU batching happy.

## Load-balancing loss

Without help, the gate collapses to a favourite expert. **Auxiliary loss**
(Fedus 2022):

```
L_lb = E · Σ_e (frac_e · mean_gate_e)
```

`frac_e` = fraction of tokens routed to expert e; `mean_gate_e` = mean gate
probability for expert e. Under uniform routing, `L_lb = k` — a strong
regulariser. Recent work (DeepSeek 2024) shows auxiliary-loss-free routing.

## Advantages

- **Parameter count scales without compute** — Mixtral-8×7B has 47B params but 13B active per token.
- **Specialisation** — different experts capture different token clusters.
- **Composition** — the same MLP block sees different weights per token.

## Costs

- **All-to-all communication** — routing tokens across GPUs is expensive; efficient MoE needs custom kernels (MegaBlocks, Tutel).
- **Load imbalance** — even with the aux loss, hotspots hurt throughput.
- **Memory** — total params must fit somewhere (usually sharded).

## When to use

- **Very large LLMs** — Mixtral, DeepSeek-V2, Grok, Gemini-Pro, Qwen2-MoE.
- **Multi-task / multi-domain models** — specialise experts to domains.
- **Compute-constrained** — an MoE inference costs the same as a dense model 4–8× smaller with (often) comparable quality.
- **Not typically** useful for small models (<1B params) — routing overhead dominates.

## Files

- `python/mixture_of_experts.py` — from-scratch MoE layer with 4 experts, top-2 routing, softmax gate, per-expert 2-layer MLP, and Switch-Transformer load-balance loss. Demo (N=200, d_in=8, d_out=4):
  - Output shape preserved.
  - Exactly 2 experts activated per token.
  - Gate rows sum to 1.0.
  - Load balance across experts 47.5–55.5% (near uniform).
  - Load-balance loss 1.99 ≈ k = 2 (uniform target).
- `r/mixture_of_experts.R` — `torch` (manual); Python `fairscale.nn.MOELayer`, `DeepSpeed-MoE`, `MegaBlocks`, `tutel`; production models Switch Transformer, Mixtral, DeepSeek-V2, Grok-1.

## Assumptions & caveats

- **Gate collapse** — without the aux loss, one expert eats all tokens; add BatchNorm on gate logits or add noise (Shazeer 2017) for stability.
- **Routing decisions are hard (top-k)** — non-differentiable; Straight-Through Estimator or Sinkhorn balancing helps.
- **Distributed all-to-all** — the practical bottleneck. Use MegaBlocks / Tutel; naive PyTorch is slow.
- **Expert-choice routing** (Zhou 2022) — reverse the roles: each expert picks its top-k tokens. Removes load-balance loss.
- **Auxiliary-loss-free** (DeepSeek 2024) — routing bias term updated at gradient time; simpler.
- **Not always better than dense** — some benchmarks prefer a same-FLOPs dense model. Iso-compute comparison is the honest one.

## Related in this repo

- `transformer-encoder`, `transformer-decoder` — the standard architecture where MoE replaces the FFN sublayer.
- `deep-mlp-backprop`, `residual-connections`, `adam-optimizer` — training-loop pairings.
- `knowledge-distillation` — a compression alternative when MoE is too expensive to serve.

## Run

```
python techniques/mixture-of-experts/python/mixture_of_experts.py
Rscript techniques/mixture-of-experts/r/mixture_of_experts.R
```

**Refs:** Shazeer, N. et al. "Outrageously large neural networks: the sparsely-gated mixture-of-experts layer." *ICLR*, 2017; Fedus, W., Zoph, B. & Shazeer, N. "Switch Transformer: scaling to trillion parameter models with simple and efficient sparsity." *JMLR* 23, 1–39, 2022; DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." *arXiv:2405.04434*, 2024.

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
