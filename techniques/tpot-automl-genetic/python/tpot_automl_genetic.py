"""TPOT - Tree-based Pipeline Optimization (Reference Sec 47.244).

Olson & Moore 2016 'TPOT: A Tree-Based Pipeline Optimization Tool
for Automating Data Science', ICML AutoML. Uses GENETIC PROGRAMMING
to search over full sklearn pipelines:

    individual = (preprocessor -> feature-selection -> ... -> estimator)
    fitness = cross-validated score
    mutations: swap components, tune hyperparameters
    crossover: exchange sub-pipelines

After N generations, TPOT emits Python code for the best pipeline.
Slower than AutoGluon's stack but explores structural diversity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


PIPELINE_SPACE = {
    "scaler": ["none", "standard", "minmax"],
    "selector": ["none", "top_k_variance"],
    "estimator": ["ridge", "rf", "gbm", "knn"],
    "n_features": [3, 5, 10],
}


def random_pipeline(rng):
    return {k: rng.choice(v) for k, v in PIPELINE_SPACE.items()}


def evaluate_pipeline(pipe, X, y, cv=3, seed=0):
    """Fit + cross-validate a pipeline; return mean MSE."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    steps = []
    if pipe["scaler"] == "standard":
        steps.append(("scaler", StandardScaler()))
    elif pipe["scaler"] == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    if pipe["selector"] == "top_k_variance":
        k = min(int(pipe["n_features"]), X.shape[1])
        steps.append(("select", SelectKBest(f_regression, k=k)))
    est = {"ridge": Ridge(),
             "rf": RandomForestRegressor(n_estimators=30, random_state=seed),
             "gbm": GradientBoostingRegressor(n_estimators=30, random_state=seed),
             "knn": KNeighborsRegressor(n_neighbors=5)}[pipe["estimator"]]
    steps.append(("est", est))
    p = Pipeline(steps)
    scores = -cross_val_score(p, X, y, cv=cv, scoring="neg_mean_squared_error")
    return float(np.mean(scores))


def genetic_search(X, y, n_gen=5, pop=8, seed=0):
    """Very toy GP: keep top half, mutate to fill."""
    rng = np.random.default_rng(seed)
    population = [random_pipeline(rng) for _ in range(pop)]
    for g in range(n_gen):
        fitness = [(evaluate_pipeline(p, X, y), p) for p in population]
        fitness.sort(key=lambda x: x[0])
        top = [p for _, p in fitness[:pop // 2]]
        # Mutate top to fill population
        new_pop = list(top)
        while len(new_pop) < pop:
            parent = rng.choice(top)
            child = dict(parent)
            k_mut = rng.choice(list(PIPELINE_SPACE.keys()))
            child[k_mut] = rng.choice(PIPELINE_SPACE[k_mut])
            new_pop.append(child)
        population = new_pop
    return fitness[0]


if __name__ == "__main__":
    print("=== TPOT (Olson-Moore 2016 ICML) ===\n")
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    best_mse, best_pipe = genetic_search(X, y, n_gen=4, pop=6, seed=0)
    print(f"  Diabetes regression, {len(y)} samples, {X.shape[1]} features")
    print(f"  Best pipeline found by genetic search (4 gens x pop 6):")
    for k, v in best_pipe.items():
        print(f"    {k:>10s} : {v}")
    print(f"  Best 3-fold CV MSE: {best_mse:.1f}")

    # Compare with ridge baseline (no pipeline search)
    baseline_mse = evaluate_pipeline({"scaler": "none", "selector": "none",
                                            "estimator": "ridge", "n_features": 10}, X, y)
    print(f"\n  Baseline ridge (no search) CV MSE: {baseline_mse:.1f}")
    print(f"  Search improvement: {100 * (1 - best_mse / baseline_mse):.1f}%")

    print("\n--- library cross-check (tpot Python; auto-sklearn; hyperopt-sklearn) ---")
