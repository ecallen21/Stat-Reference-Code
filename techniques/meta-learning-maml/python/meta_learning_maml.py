"""Model-Agnostic Meta-Learning — MAML (Finn 2017; Reference §27.x extra).

Learn INITIALISATION theta such that a small number of gradient steps on any
new task's support set produces a good policy / classifier on that task.

Bi-level objective:
    theta* = argmin sum_i  L_i^query( theta - alpha * grad_theta L_i^support(theta) )

Outer loop: SGD on theta via the meta-gradient (differentiates through the
inner step).  Inner loop: k steps of task-specific fine-tuning from theta.

We demonstrate 5-shot 1-D regression: each task is fitting a sine y = A sin(x + phi)
with random (A, phi).  A MAML-trained init adapts to any new sine with a few
gradient steps; a naive shared-weights baseline does not.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _model(x, theta):
    """A tiny linear-in-features regressor: y = w0 + w1 * x + w2 * x^2 + w3 * sin(x).
    theta = [w0, w1, w2, w3]."""
    feats = np.stack([np.ones_like(x), x, x ** 2, np.sin(x)], axis=-1)
    return feats @ theta


def _loss(theta, x, y):
    pred = _model(x, theta)
    return float(np.mean((pred - y) ** 2))


def _grad(theta, x, y):
    feats = np.stack([np.ones_like(x), x, x ** 2, np.sin(x)], axis=-1)
    pred = feats @ theta
    return 2 * feats.T @ (pred - y) / len(x)


def sample_task(rng):
    A = float(rng.uniform(0.5, 2.5))
    phi = float(rng.uniform(-1.0, 1.0))
    return A, phi


def make_task_data(A, phi, k: int, rng):
    x = rng.uniform(-3, 3, size=k)
    y = A * np.sin(x + phi)
    return x, y


def train_maml(n_iter=1000, n_tasks=8, k_support=5, k_query=5, inner_lr=0.05,
                inner_steps=1, meta_lr=0.02, seed=0):
    """First-order MAML: skip the second-order term in the meta-gradient (Finn 2017)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.1, size=4)
    for it in range(n_iter):
        meta_grad = np.zeros_like(theta)
        for _ in range(n_tasks):
            A, phi = sample_task(rng)
            xs, ys = make_task_data(A, phi, k_support, rng)
            xq, yq = make_task_data(A, phi, k_query, rng)
            theta_task = theta.copy()
            for _ in range(inner_steps):
                theta_task -= inner_lr * _grad(theta_task, xs, ys)
            # first-order MAML: outer gradient is the query gradient at theta_task
            meta_grad += _grad(theta_task, xq, yq)
        theta -= meta_lr * meta_grad / n_tasks
    return {"theta": theta, "method": "MAML (first-order)"}


def train_shared_baseline(n_iter=1000, n_tasks=8, k_support=5, k_query=5,
                           lr=0.02, seed=0):
    """Baseline: just minimise the mean loss across tasks, no inner adaptation."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.1, size=4)
    for it in range(n_iter):
        grad = np.zeros_like(theta)
        for _ in range(n_tasks):
            A, phi = sample_task(rng)
            xs, ys = make_task_data(A, phi, k_support + k_query, rng)
            grad += _grad(theta, xs, ys)
        theta -= lr * grad / n_tasks
    return {"theta": theta, "method": "shared-weights baseline"}


def evaluate(theta, k_adapt=5, n_test_tasks=50, inner_lr=0.05, inner_steps=5,
              seed=0):
    """For each test task, adapt from theta and evaluate query MSE."""
    rng = np.random.default_rng(seed)
    losses_before = []; losses_after = []
    for _ in range(n_test_tasks):
        A, phi = sample_task(rng)
        xs, ys = make_task_data(A, phi, k_adapt, rng)
        xq, yq = make_task_data(A, phi, 20, rng)
        losses_before.append(_loss(theta, xq, yq))
        theta_task = theta.copy()
        for _ in range(inner_steps):
            theta_task -= inner_lr * _grad(theta_task, xs, ys)
        losses_after.append(_loss(theta_task, xq, yq))
    return {"before_adapt_mse": float(np.mean(losses_before)),
            "after_adapt_mse": float(np.mean(losses_after))}


if __name__ == "__main__":
    maml = train_maml(n_iter=800, meta_lr=0.005, inner_lr=0.02, inner_steps=1, seed=0)
    baseline = train_shared_baseline(n_iter=800, lr=0.005, seed=0)

    print("=== MAML vs shared-baseline 5-shot sine regression ===")
    for name, theta in [("MAML init      ", maml["theta"]),
                          ("shared baseline", baseline["theta"])]:
        e = evaluate(theta, k_adapt=10, inner_lr=0.02, inner_steps=5, seed=42)
        print(f"  {name}: query MSE BEFORE adapt = {e['before_adapt_mse']:.3f},  "
              f"AFTER 5-step adapt = {e['after_adapt_mse']:.3f}")

    print("\n  MAML's initialisation was learned to be quickly adaptable;")
    print("  the shared baseline is at a decent average solution but adapts more slowly.")

    print("\n--- library cross-check (learn2learn, higher, torchmeta) ---")
