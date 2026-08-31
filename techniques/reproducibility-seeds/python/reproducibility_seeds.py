"""Reproducibility: seeds + hash checks (Reference Ch 32 MLOps).

For a training run to be REPRODUCIBLE, three things must be pinned:

  1. All RNG SEEDS  (python random, numpy, torch, cuda, framework-specific).
  2. Deterministic OPS  (torch.use_deterministic_algorithms; disable
     nondeterministic CUDA kernels; set env vars like PYTHONHASHSEED,
     CUBLAS_WORKSPACE_CONFIG).
  3. Data + code + environment PROVENANCE HASHES (git commit, dataset
     SHA-256, pip freeze lockfile hash).

Here we implement:
  * seed_everything(seed)                    -- one call to seed common RNGs.
  * hash_bytes(x), hash_array(np.ndarray)    -- SHA-256 fingerprints.
  * reproducibility_manifest()               -- machine-readable capture.
  * A demo showing that two independent runs seeded with the same seed
    produce IDENTICAL model weights (same SHA-256 of the final weights),
    while unseeded runs diverge.
"""
from __future__ import annotations    # stdlib

import hashlib   # SHA-256
import os        # env vars
import platform  # OS / arch capture
import random    # python RNG
import sys       # python version

import numpy as np    # numerical + numpy RNG


def seed_everything(seed):
    """Seed Python + numpy RNGs and set PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)      # if torch available
    return seed


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_array(a: np.ndarray) -> str:
    return hash_bytes(a.tobytes())


def reproducibility_manifest(seed):
    return {
        "seed": int(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def train_toy_model(seed, n_iters=200, n_features=5):
    """Toy SGD; returns final weight vector."""
    seed_everything(seed)
    X = np.random.normal(0, 1, (300, n_features))
    beta_true = np.random.normal(0, 1, n_features)
    y = X @ beta_true + np.random.normal(0, 0.5, 300)
    beta = np.zeros(n_features)
    for _ in range(n_iters):
        idx = np.random.integers(0, len(X), 32) if hasattr(np.random, "integers") else np.random.randint(0, len(X), 32)
        Xb, yb = X[idx], y[idx]
        g = 2 * Xb.T @ (Xb @ beta - yb) / len(yb)
        beta -= 0.01 * g
    return beta


if __name__ == "__main__":
    print("=== Reproducibility: seeds + provenance hashes ===\n")
    # 1) Manifest capture (after seed_everything so PYTHONHASHSEED is set).
    seed_everything(42)
    m = reproducibility_manifest(seed=42)
    print("  Reproducibility manifest:")
    for k, v in m.items():
        print(f"    {k:20s}   {v}")

    # 2) Two runs, same seed -> identical outputs.
    b1 = train_toy_model(seed=42)
    b2 = train_toy_model(seed=42)
    h1, h2 = hash_array(b1), hash_array(b2)
    print(f"\n  Seed 42, run 1 weight SHA-256: {h1[:16]}...")
    print(f"  Seed 42, run 2 weight SHA-256: {h2[:16]}...")
    print(f"  Match: {h1 == h2}\n")

    # 3) Different seed -> different weights.
    b3 = train_toy_model(seed=99)
    h3 = hash_array(b3)
    print(f"  Seed 99 weight SHA-256:        {h3[:16]}...")
    print(f"  Same as seed 42? {h1 == h3}\n")

    # 4) Dataset hash (a common provenance stamp).
    dataset_bytes = b"synthetic-30k-transactions-2025-08-31"
    print(f"  Dataset SHA-256: {hash_bytes(dataset_bytes)[:16]}...\n")

    print("--- library cross-check (pytorch-lightning.seed_everything;"
          " transformers.set_seed; dvc; reprozip) ---")
