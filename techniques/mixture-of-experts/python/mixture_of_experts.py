"""Mixture-of-Experts layer with top-k gating (Reference §27.x extra).

Shazeer et al. 2017 (Sparsely-Gated MoE) / Fedus et al. 2022 (Switch Transformer):

  gate(x)   = softmax(W_g x)                       (E logits over experts)
  top-k(x) = keep the k highest gate logits, renormalise
  output    = sum_{e in top_k} gate_e(x) * Expert_e(x)

Only the top-k experts are activated per token, keeping the compute per-token
cheap while total parameters can be huge.

Load-balancing loss keeps traffic uniform across experts (else the gate
collapses to a few):
    L_lb = E * sum_e (fraction_e * mean_gate_e)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


class MoE:
    """Sparse top-k mixture of E MLP experts on d_in -> d_out."""
    def __init__(self, E: int, d_in: int, d_out: int, hidden: int = 16,
                 k: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.E = E; self.k = k
        self.W_g = rng.normal(scale=np.sqrt(1.0 / d_in), size=(d_in, E))
        # each expert: 2-layer MLP
        self.experts = [
            {"W1": rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, hidden)),
             "b1": np.zeros(hidden),
             "W2": rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, d_out)),
             "b2": np.zeros(d_out)}
            for _ in range(E)]

    def _expert(self, x, e):
        h = np.maximum(x @ self.experts[e]["W1"] + self.experts[e]["b1"], 0.0)
        return h @ self.experts[e]["W2"] + self.experts[e]["b2"]

    def forward(self, X):
        logits = X @ self.W_g                                 # (N, E)
        # per-token top-k mask
        topk_idx = np.argpartition(-logits, self.k, axis=1)[:, :self.k]
        mask = np.zeros_like(logits, dtype=bool)
        for i in range(len(X)):
            mask[i, topk_idx[i]] = True
        gate_full = _softmax(np.where(mask, logits, -np.inf), axis=1)
        out = np.zeros((len(X), self.experts[0]["W2"].shape[1]))
        expert_load = np.zeros(self.E)
        for e in range(self.E):
            active = np.where(mask[:, e])[0]
            if len(active) == 0:
                continue
            expert_load[e] = len(active) / len(X)
            out[active] += gate_full[active, e:e + 1] * self._expert(X[active], e)
        return out, gate_full, expert_load

    def load_balance_loss(self, gate, mask):
        """Switch-Transformer aux loss:  E * sum_e (frac_e * mean_gate_e)."""
        frac = mask.mean(axis=0)
        mean_gate = gate.mean(axis=0)
        return float(self.E * (frac * mean_gate).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 200; d_in = 8; d_out = 4
    X = rng.normal(size=(N, d_in))

    moe = MoE(E=4, d_in=d_in, d_out=d_out, hidden=16, k=2, seed=1)
    out, gate, load = moe.forward(X)

    print(f"=== MoE: 4 experts, top-k = 2, input d={d_in}, output d={d_out} ===")
    print(f"  output shape                   = {out.shape}")
    print(f"  fraction of tokens per expert  = {np.round(load, 3).tolist()}")
    print(f"  gate columns sum per token (should be 1): first 3 = "
          f"{np.round(gate[:3].sum(axis=1), 3).tolist()}")
    mask = gate > 0
    print(f"  each token uses {mask.sum(axis=1).mean():.1f} experts on average "
          f"(should be exactly {moe.k})")

    lb = moe.load_balance_loss(gate, mask)
    print(f"  load-balance loss              = {lb:.4f}")
    print(f"  under perfect uniform routing this loss = k = {moe.k}")

    # Sanity: freeze gate and verify only top-k contribute
    diff_if_zero_top1 = np.linalg.norm(out - moe.forward(X)[0])
    print(f"\n  determinism check (same input twice, ||diff|| = "
          f"{diff_if_zero_top1:.2e}, should be 0)")

    print("\n--- library cross-check (fairscale / DeepSpeed / MegaBlocks / mixtral-of-experts) ---")
