"""SimCLR-style contrastive representation learning (Reference §27.x extra).

Given batches of N examples, each with two augmented "views" (x_i, x_i'), map
each view through an encoder f + projection head g to a unit vector z:

    z_i = g(f(x_i)) / ||g(f(x_i))||

NT-Xent (Chen et al. 2020) InfoNCE loss with temperature tau:

    L_i = -log( exp(sim(z_i, z_i') / tau)
                / sum_{k != i} exp(sim(z_i, z_k) / tau + z_k') / tau) )

We use a minimal encoder (linear projection to R^d), unit-normalise, and
train with a bounded contrastive loss.  Demo: two identity-preserving
"views" of a set of latent codes; after training, pairs land in nearby
directions of the embedding space.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _l2_normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def train_simclr(views1, views2, dim: int = 8, tau: float = 0.2,
                  lr: float = 0.05, epochs: int = 300, seed: int = 0) -> dict:
    """views1, views2: (N, d_input); each pair is a positive; all other pairs are negatives.
    Manual gradient (approximate) via numerical for-scheme kept short for demo."""
    rng = np.random.default_rng(seed)
    v1 = np.asarray(views1, dtype=float); v2 = np.asarray(views2, dtype=float)
    N, d_in = v1.shape
    W = rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, dim))

    def _forward(V):
        return _l2_normalise(V @ W)

    def _loss_and_grad(W):
        z1 = _forward(v1); z2 = _forward(v2)
        # NT-Xent: pair (i, i) positive, all other cross- and same-view rows are negatives.
        # Combine into a 2N-by-2N similarity matrix
        Z = np.vstack([z1, z2])                                   # (2N, dim)
        sim = Z @ Z.T / tau                                        # (2N, 2N)
        np.fill_diagonal(sim, -np.inf)                             # exclude self
        # positives are (i, N+i) and (N+i, i)
        labels = np.concatenate([np.arange(N, 2 * N), np.arange(N)])
        # softmax cross-entropy over rows
        sim_max = sim.max(axis=1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        exp_sim[np.arange(2 * N), np.arange(2 * N)] = 0.0
        Z_sum = exp_sim.sum(axis=1)
        log_prob = sim[np.arange(2 * N), labels] - sim_max.squeeze() - np.log(Z_sum + 1e-12)
        loss = -log_prob.mean()
        # crude finite-diff gradient (small enough for the demo)
        grad = np.zeros_like(W)
        h = 1e-4
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += h
                z1p = _forward(v1); z2p = _forward(v2)
                Zp = np.vstack([z1p, z2p])
                simp = Zp @ Zp.T / tau
                np.fill_diagonal(simp, -np.inf)
                mp = simp.max(axis=1, keepdims=True)
                ep = np.exp(simp - mp)
                ep[np.arange(2 * N), np.arange(2 * N)] = 0.0
                Zsp = ep.sum(axis=1)
                lp = simp[np.arange(2 * N), labels] - mp.squeeze() - np.log(Zsp + 1e-12)
                loss_p = -lp.mean()
                grad[i, j] = (loss_p - loss) / h
                W[i, j] -= h
        return loss, grad

    losses = []
    for ep in range(epochs):
        loss, grad = _loss_and_grad(W)
        W -= lr * grad
        losses.append(loss)
    return {"W": W, "losses": losses, "tau": tau,
            "method": "SimCLR NT-Xent contrastive (numerical grad)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Latent identities: N = 15 anchor points in R^6
    N = 15
    latents = rng.normal(size=(N, 6))
    # Two "augmented views": add small noise
    v1 = latents + 0.15 * rng.normal(size=latents.shape)
    v2 = latents + 0.15 * rng.normal(size=latents.shape)

    m = train_simclr(v1, v2, dim=4, tau=0.2, lr=0.02, epochs=40)
    z1 = _l2_normalise(v1 @ m["W"]); z2 = _l2_normalise(v2 @ m["W"])
    # per-pair cosine similarity (should be high) vs mean cross-pair (should be low)
    pos = [z1[i] @ z2[i] for i in range(N)]
    neg = [z1[i] @ z2[j] for i in range(N) for j in range(N) if i != j]
    print(f"=== SimCLR-style contrastive on toy views (N={N}) ===")
    print(f"  final NT-Xent loss = {m['losses'][-1]:.4f}   (initial {m['losses'][0]:.4f})")
    print(f"  mean POSITIVE cosine sim (matched view pairs)   = {np.mean(pos):+.3f}")
    print(f"  mean NEGATIVE cosine sim (cross-pair)           = {np.mean(neg):+.3f}")

    # rank of retrieval: for each z1_i, is z2_i the nearest?
    S = z1 @ z2.T
    ranks = []
    for i in range(N):
        r = int((-S[i]).argsort().tolist().index(i)) + 1
        ranks.append(r)
    print(f"  mean rank of the true pair (1 = perfect): {np.mean(ranks):.2f}")
    print(f"  fraction @rank-1: {(np.array(ranks) == 1).mean():.3f}")

    print("\n--- library cross-check (torch SimCLR / MoCo / SimSiam / DINO / lightly) ---")
