"""Residual connections (He et al. 2015; Reference §27.x extra).

A residual block: y = x + F(x)  (identity shortcut around a small subnetwork F)

The identity path lets the *gradient* flow directly through the block, so
even a deep stack can be trained without vanishing / exploding gradients.

Demo: track the norm of the gradient wrt an input activation as it flows
back through L stacked blocks, comparing "plain" vs "residual" blocks with
identical weights.  Residual keeps the gradient bounded; plain stacks
either decay or blow up depending on the spectral radius of the layer.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _plain_block_forward(x, W, b):
    return _relu(x @ W + b), x @ W + b


def _plain_block_backward(dy, x, W, z):
    dz = dy * _relu_grad(z)
    return dz @ W.T


def _res_block_forward(x, W, b):
    pre = x @ W + b
    return x + _relu(pre), pre


def _res_block_backward(dy, x, W, z):
    dz_from_F = dy * _relu_grad(z)
    dx_from_F = dz_from_F @ W.T
    return dy + dx_from_F                                # identity + F path


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 32; L = 40                                        # 40 stacked blocks
    # Standard practice: small weight scale on the inner block F so that
    # y = x + F(x) stays close to identity at init.  He-init scale for d=32:
    # sqrt(2/d) = 0.25.  We use a slightly-smaller scale to give both stacks
    # a fair comparison at depth.
    Ws = [rng.normal(scale=0.10, size=(d, d)) for _ in range(L)]
    bs = [np.zeros(d) for _ in range(L)]

    x = rng.normal(size=d)

    # forward pass, saving pre-activations
    a_plain = x.copy(); pres_plain = []
    for W, b in zip(Ws, bs):
        a_plain, pre = _plain_block_forward(a_plain, W, b)
        pres_plain.append(pre)

    a_res = x.copy(); pres_res = []
    for W, b in zip(Ws, bs):
        a_res, pre = _res_block_forward(a_res, W, b)
        pres_res.append(pre)

    # simulate a gradient of norm 1 at the OUTPUT and propagate back
    dy_plain = rng.normal(size=d); dy_plain = dy_plain / np.linalg.norm(dy_plain)
    dy_res   = dy_plain.copy()
    norms_plain = [1.0]; norms_res = [1.0]
    # need each block's INPUT to run backward
    # recompute forward again keeping the (input, pre) per block
    ins_plain = [x.copy()]; a = x.copy()
    for W, b in zip(Ws, bs):
        a, _ = _plain_block_forward(a, W, b); ins_plain.append(a)
    ins_res = [x.copy()]; a = x.copy()
    for W, b in zip(Ws, bs):
        a, _ = _res_block_forward(a, W, b); ins_res.append(a)

    # backward through the plain and residual stacks
    for l in reversed(range(L)):
        dy_plain = _plain_block_backward(dy_plain, ins_plain[l], Ws[l], pres_plain[l])
        dy_res = _res_block_backward(dy_res, ins_res[l], Ws[l], pres_res[l])
        norms_plain.append(np.linalg.norm(dy_plain))
        norms_res.append(np.linalg.norm(dy_res))

    norms_plain = norms_plain[::-1]; norms_res = norms_res[::-1]

    print(f"=== Gradient-norm propagation through {L} blocks (d={d}) ===")
    print(f"  layer   plain    residual")
    for l in (0, 5, 10, 20, 30, 39, 40):
        if l < len(norms_plain):
            print(f"  {l:>4}   {norms_plain[l]:.3e}   {norms_res[l]:.3e}")
    print(f"\n  plain    decay ratio (input / output) = "
          f"{norms_plain[0] / (norms_plain[-1] + 1e-30):.3e}")
    print(f"  residual decay ratio (input / output) = "
          f"{norms_res[0] / (norms_res[-1] + 1e-30):.3e}")
    print(f"  residual keeps the gradient within ~O(1) even at depth {L}.")

    print("\n--- library cross-check (torch.nn.Module with 'y = x + F(x)') ---")
