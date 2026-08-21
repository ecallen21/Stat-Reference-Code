"""LSTM and GRU cells (Reference §27.4).

LSTM (Hochreiter-Schmidhuber 1997):
    i_t = sigma(W_i [x_t; h_{t-1}] + b_i)      input gate
    f_t = sigma(W_f [x_t; h_{t-1}] + b_f)      forget gate
    o_t = sigma(W_o [x_t; h_{t-1}] + b_o)      output gate
    g_t = tanh(W_g [x_t; h_{t-1}] + b_g)       candidate
    c_t = f_t * c_{t-1} + i_t * g_t             cell state (additive path)
    h_t = o_t * tanh(c_t)                        hidden state

GRU (Cho et al. 2014):
    z_t = sigma(W_z [x_t; h_{t-1}])             update gate
    r_t = sigma(W_r [x_t; h_{t-1}])             reset gate
    h_tilde = tanh(W_h [x_t; r_t * h_{t-1}])
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde   convex mix

Both alleviate vanishing gradients via a mostly-additive cell path (LSTM's
c_t; GRU's convex mix).  We implement the forward passes here and demonstrate
that both propagate a signal across 30 time-steps that a plain RNN loses.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _init(rng, shape, scale=0.1):
    return rng.normal(scale=scale, size=shape)


def lstm_forward(X, W, U, b, h0, c0):
    """X: (T, d_in);  W: (4, d_in, d_h);  U: (4, d_h, d_h);  b: (4, d_h).
    order: i, f, g, o."""
    T = X.shape[0]; d_h = h0.shape[0]
    h = h0.copy(); c = c0.copy()
    hs = [h.copy()]; cs = [c.copy()]
    for t in range(T):
        x = X[t]
        i = _sigmoid(x @ W[0] + h @ U[0] + b[0])
        f = _sigmoid(x @ W[1] + h @ U[1] + b[1])
        g = np.tanh(x @ W[2] + h @ U[2] + b[2])
        o = _sigmoid(x @ W[3] + h @ U[3] + b[3])
        c = f * c + i * g
        h = o * np.tanh(c)
        hs.append(h.copy()); cs.append(c.copy())
    return hs, cs


def gru_forward(X, W, U, b, h0):
    """X: (T, d_in);  W: (3, d_in, d_h);  U: (3, d_h, d_h);  b: (3, d_h).
    order: z, r, h_tilde."""
    T = X.shape[0]; d_h = h0.shape[0]
    h = h0.copy(); hs = [h.copy()]
    for t in range(T):
        x = X[t]
        z = _sigmoid(x @ W[0] + h @ U[0] + b[0])
        r = _sigmoid(x @ W[1] + h @ U[1] + b[1])
        h_tilde = np.tanh(x @ W[2] + (r * h) @ U[2] + b[2])
        h = (1 - z) * h + z * h_tilde
        hs.append(h.copy())
    return hs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 30; d_in = 4; d_h = 8

    # Plain RNN cell forward pass, for contrast: multiplicative decay of an initial spike.
    x_seq = np.zeros((T, d_in)); x_seq[0, 0] = 1.0        # spike only at t=0
    Wx_r = _init(rng, (d_in, d_h)); Wh_r = _init(rng, (d_h, d_h))
    b_r = np.zeros(d_h)
    h_rnn = np.zeros(d_h); norms_rnn = [np.linalg.norm(h_rnn)]
    for t in range(T):
        h_rnn = np.tanh(x_seq[t] @ Wx_r + h_rnn @ Wh_r + b_r)
        norms_rnn.append(np.linalg.norm(h_rnn))

    # LSTM
    W_l = _init(rng, (4, d_in, d_h)); U_l = _init(rng, (4, d_h, d_h))
    b_l = np.zeros((4, d_h)); b_l[1] = 1.0                # forget bias = 1 (standard trick)
    hs_l, cs_l = lstm_forward(x_seq, W_l, U_l, b_l, np.zeros(d_h), np.zeros(d_h))
    norms_lstm = [np.linalg.norm(h) for h in hs_l]

    # GRU
    W_g = _init(rng, (3, d_in, d_h)); U_g = _init(rng, (3, d_h, d_h))
    b_g = np.zeros((3, d_h))
    hs_g = gru_forward(x_seq, W_g, U_g, b_g, np.zeros(d_h))
    norms_gru = [np.linalg.norm(h) for h in hs_g]

    print(f"=== Hidden-state norm across T={T} steps after a t=0 input spike ===")
    print(f"  t   RNN     LSTM    GRU")
    for t in (0, 1, 5, 10, 20, 30):
        print(f"  {t:>2}   {norms_rnn[t]:.4f}  {norms_lstm[t]:.4f}  {norms_gru[t]:.4f}")
    print(f"\n  RNN decay ratio (t=30 / t=1): {norms_rnn[30] / (norms_rnn[1] + 1e-12):.3e}")
    print(f"  LSTM decay ratio             : {norms_lstm[30] / (norms_lstm[1] + 1e-12):.3e}")
    print(f"  (LSTM's cell path preserves signal much longer.)")

    print("\n--- library cross-check (torch.nn.LSTM / GRU) ---")
    try:
        import torch
        import torch.nn as nn
        Xt = torch.tensor(x_seq[None], dtype=torch.float32)   # (1, T, d_in)
        lstm = nn.LSTM(d_in, d_h, batch_first=True)
        gru = nn.GRU(d_in, d_h, batch_first=True)
        with torch.no_grad():
            out_l, _ = lstm(Xt); out_g, _ = gru(Xt)
            print(f"  torch LSTM final |h| = {out_l[0, -1].norm().item():.4f}")
            print(f"  torch GRU  final |h| = {out_g[0, -1].norm().item():.4f}")
    except ImportError:
        print("  (pytorch not installed)")
