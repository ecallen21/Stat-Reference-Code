"""Deep Q-Network (Mnih 2013, 2015; Reference §28.3).

Neural-net approximator to Q(s, a) with two critical stabilisers:

  1. EXPERIENCE REPLAY: store (s, a, r, s', done) tuples in a buffer; sample
     minibatches to break temporal correlation.
  2. TARGET NETWORK: a copy of Q with weights frozen for N steps; used for
     the TD target y = r + gamma * max_a' Q_target(s', a') to prevent the
     bootstrapping instability.

Loss: y - Q(s, a) squared, updated via gradient step.

Demo: small MLP Q approximator on a 1D LineWorld with 5 states + 2 actions.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import deque    # stdlib: bounded FIFO buffer

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _forward(s_onehot, W1, b1, W2, b2):
    z = _relu(s_onehot @ W1 + b1)
    return z @ W2 + b2, z


class DQN:
    def __init__(self, n_states, n_actions, hidden=16, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n_states = n_states; self.n_actions = n_actions; self.lr = lr
        self.W1 = rng.normal(scale=np.sqrt(2.0 / n_states), size=(n_states, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        # target-network copy
        self.tW1, self.tb1, self.tW2, self.tb2 = (self.W1.copy(), self.b1.copy(),
                                                    self.W2.copy(), self.b2.copy())
        self.buffer = deque(maxlen=1000)

    def sync_target(self):
        self.tW1[:] = self.W1; self.tb1[:] = self.b1
        self.tW2[:] = self.W2; self.tb2[:] = self.b2

    def _onehot(self, s):
        v = np.zeros(self.n_states); v[s] = 1.0; return v

    def q_values(self, s):
        return _forward(self._onehot(s), self.W1, self.b1, self.W2, self.b2)[0]

    def _q_batch(self, states, W1, b1, W2, b2):
        X = np.zeros((len(states), self.n_states))
        X[np.arange(len(states)), states] = 1.0
        z = _relu(X @ W1 + b1)
        return z @ W2 + b2, z, X

    def update(self, batch, gamma=0.9):
        s, a, r, s2, done = zip(*batch)
        s = np.array(s); a = np.array(a); r = np.array(r, dtype=float)
        s2 = np.array(s2); done = np.array(done, dtype=float)
        q_next_all, _, _ = self._q_batch(s2, self.tW1, self.tb1, self.tW2, self.tb2)
        y = r + gamma * q_next_all.max(axis=1) * (1 - done)
        q_all, z, X = self._q_batch(s, self.W1, self.b1, self.W2, self.b2)
        y_pred = q_all[np.arange(len(batch)), a]
        # gradient of 0.5 (y_pred - y)^2 wrt output for action a only
        dq = np.zeros_like(q_all)
        dq[np.arange(len(batch)), a] = (y_pred - y) / len(batch)
        dW2 = z.T @ dq; db2 = dq.sum(axis=0)
        dz = dq @ self.W2.T
        dz_pre = dz * _relu_grad(X @ self.W1 + self.b1)
        dW1 = X.T @ dz_pre; db1 = dz_pre.sum(axis=0)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


class LineWorld:
    def __init__(self, n=5): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


if __name__ == "__main__":
    env = LineWorld(n=5)
    dqn = DQN(n_states=5, n_actions=2, hidden=8, lr=0.05, seed=0)
    rng = np.random.default_rng(0)
    n_ep = 200; batch = 32; sync_every = 20; eps = 0.2
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0
        while not done:
            a = int(rng.integers(2)) if rng.uniform() < eps else int(dqn.q_values(s).argmax())
            s2, r, done = env.step(s, a)
            dqn.buffer.append((s, a, r, s2, done))
            s = s2; total += r
            if len(dqn.buffer) >= batch:
                idx = rng.choice(len(dqn.buffer), batch, replace=False)
                b = [dqn.buffer[i] for i in idx]
                dqn.update(b, gamma=0.9)
        if ep % sync_every == 0:
            dqn.sync_target()
        returns.append(total)

    print("=== DQN on LineWorld (5 states, 2 actions) ===")
    print(f"  final Q-values by state (left, right):")
    for s in range(env.n):
        print(f"    state {s}: {np.round(dqn.q_values(s), 2).tolist()}")
    print(f"  greedy policy: "
          f"{[('R' if int(dqn.q_values(s).argmax()) == 1 else 'L') for s in range(env.n)]}")
    print(f"  mean return over last 20 episodes: {np.mean(returns[-20:]):.2f}   "
          f"(optimal 7.0 = 10 - 3 steps of -1)")

    print("\n--- library cross-check (stable-baselines3 DQN; cleanrl/dqn.py) ---")
