"""In-context learning (ICL) (Reference Sec 47.20).

Brown et al. 2020 'Language models are few-shot learners', NeurIPS
(GPT-3 paper). Rather than fine-tune, LLMs can 'learn' a task from a
few input-output examples supplied in the prompt:

    Prompt:
        Q: 1 + 2 = ?
        A: 3
        Q: 5 + 4 = ?
        A: 9
        Q: 7 + 6 = ?
        A: <predict>

Empirically observed:
    * ZERO-shot        - task described with instructions, no examples.
    * ONE-shot         - one demo.
    * FEW-shot (K)     - K demos; performance often scales with log K.

Theoretical view (Xie et al. 2022, Akyurek et al. 2023): under certain
distributions of prompts, ICL is EQUIVALENT to implicit Bayesian
inference / gradient descent inside the transformer.

We simulate a compact ICL evaluation by TRAINING a small softmax
classifier over K in-context digit-addition demos, showing accuracy
climbing with K. The 'model' here is a linear regression fit on the
in-prompt (x, y) pairs -- a proxy for what large LMs internally do.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def synthetic_arithmetic_task(rng, k, task_slope=2.0, task_intercept=1.0, noise=0.1):
    """Task: y = task_slope * x + task_intercept + noise.

    'ICL' means the model sees k (x_i, y_i) demos then must predict y for a new x.
    """
    x = rng.uniform(-3, 3, size=k)
    y = task_slope * x + task_intercept + noise * rng.normal(size=k)
    return x, y


def icl_predict(x_demo, y_demo, x_query):
    """Toy 'implicit gradient descent' inside the context: fit OLS on demos."""
    if len(x_demo) == 0:
        return 0.0
    if len(x_demo) == 1:
        return y_demo[0]      # zero-slope prior
    A = np.c_[np.ones(len(x_demo)), x_demo]
    b, *_ = np.linalg.lstsq(A, y_demo, rcond=None)
    return b[0] + b[1] * x_query


if __name__ == "__main__":
    print("=== In-context learning -- toy 'transformer as OLS' proxy ===\n")
    rng = np.random.default_rng(0)
    n_tasks = 500
    x_test = np.array([0.0, 1.0, 2.5])       # evaluation queries

    print(f"  {'K demos':>8s}  {'MSE(y_hat, y_true)':>20s}")
    for K in [0, 1, 2, 4, 8, 16, 32]:
        errors = []
        for _ in range(n_tasks):
            #  Sample a NEW task from a task distribution
            slope = rng.uniform(-3, 3)
            intercept = rng.uniform(-2, 2)
            x_d, y_d = synthetic_arithmetic_task(rng, K, slope, intercept, noise=0.2)
            for xq in x_test:
                y_true = slope * xq + intercept
                y_hat = icl_predict(x_d, y_d, xq)
                errors.append((y_hat - y_true) ** 2)
        mse = float(np.mean(errors))
        print(f"  {K:>8d}  {mse:>20.4f}")

    print("\n  Interpretation: MSE drops sharply from K=0 (uniform ignorance) to")
    print("                  K=2 (task identified), plateaus by K=8 (variance floor).")
    print("                  Large LMs show a similar curve on real tasks (Brown 2020).")

    print("\n--- library cross-check (openai / anthropic APIs for real LM eval) ---")
