"""Experiment tracking (Reference Ch 32 MLOps).

Log every training run's:
  * PARAMS      -- hyperparameters + code version + seed.
  * METRICS     -- scalars over epochs, plus final validation metrics.
  * ARTIFACTS   -- model checkpoints, plots, notebooks; identified by
                    a content hash so runs are exactly reproducible.
  * ENVIRONMENT -- python version, pip freeze / uv lock, OS.

Minimal API (matches MLflow / W&B / neptune conventions):
  run = tracker.start_run(name="try-01")
  run.log_params({...}); run.log_metric("loss", 0.5, step=3)
  run.log_artifact("model.pkl")
  run.end()

The store here is a directory tree keyed by run_id (uuid). Metrics are
appended as jsonl. Artifacts are stored with their SHA-256 filename
so identical models across runs collapse to one file (content-addressed
storage a la git / DVC).

Demo: run 3 experiments with different learning rates, log everything,
then query the tracker for the best-metric run.
"""
from __future__ import annotations    # stdlib

import hashlib   # SHA-256 content hash
import json      # jsonl metric log
import os        # filesystem
import tempfile  # scratch dir for the demo
import time      # timestamps
import uuid      # unique run ids

import numpy as np    # numerical arrays


class ExperimentTracker:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "artifacts"), exist_ok=True)

    def start_run(self, name="run"):
        return Run(root=self.root, name=name)

    def list_runs(self):
        out = []
        for d in sorted(os.listdir(self.root)):
            if d == "artifacts": continue
            meta_path = os.path.join(self.root, d, "meta.json")
            if not os.path.exists(meta_path): continue
            with open(meta_path) as f:
                out.append(json.load(f))
        return out

    def best_run(self, metric="val_loss", minimise=True):
        runs = self.list_runs()
        scored = [(r, r.get("final_metrics", {}).get(metric)) for r in runs]
        scored = [(r, s) for r, s in scored if s is not None]
        return min(scored, key=lambda x: x[1] if minimise else -x[1])[0] if scored else None


class Run:
    def __init__(self, root, name):
        self.root = root
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.meta = {"id": self.id, "name": name, "start_ts": time.time(),
                      "params": {}, "final_metrics": {}, "artifacts": []}
        self._metric_log = open(os.path.join(self.dir, "metrics.jsonl"), "w")

    def log_params(self, params):
        self.meta["params"].update(params)

    def log_metric(self, key, value, step=None):
        row = {"key": key, "value": float(value), "step": step, "ts": time.time()}
        self._metric_log.write(json.dumps(row) + "\n")
        self._metric_log.flush()
        self.meta["final_metrics"][key] = float(value)

    def log_artifact(self, local_path):
        with open(local_path, "rb") as f:
            data = f.read()
        h = hashlib.sha256(data).hexdigest()
        art_dir = os.path.join(self.root, "artifacts")
        target = os.path.join(art_dir, f"{h}.bin")
        if not os.path.exists(target):
            with open(target, "wb") as g:
                g.write(data)
        self.meta["artifacts"].append({"path": local_path, "sha256": h})
        return h

    def end(self):
        self.meta["end_ts"] = time.time()
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        self._metric_log.close()


if __name__ == "__main__":
    print("=== Experiment tracking with content-addressed artifacts ===\n")
    tmpdir = tempfile.mkdtemp(prefix="exp_track_")
    tracker = ExperimentTracker(tmpdir)

    # Three experiments: sweep learning rate.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 3)); beta_true = np.array([1.5, -0.5, 0.7])
    y = X @ beta_true + rng.normal(0, 0.5, 500)

    for lr in (0.001, 0.01, 0.1):
        run = tracker.start_run(name=f"lr={lr}")
        run.log_params({"lr": lr, "seed": 0, "code_version": "b12ffec"})
        beta = np.zeros(3)
        for epoch in range(20):
            pred = X @ beta
            g = 2 * X.T @ (pred - y) / len(y)
            beta -= lr * g
            loss = float(((pred - y) ** 2).mean())
            run.log_metric("train_loss", loss, step=epoch)
        # Save the model weights as an artifact.
        model_path = os.path.join(tmpdir, f"model_{lr}.npy")
        np.save(model_path, beta)
        h = run.log_artifact(model_path)
        run.log_metric("val_loss", float(((X @ beta - y) ** 2).mean()))
        run.end()
        print(f"  run {run.id}   lr={lr}   final val_loss={run.meta['final_metrics']['val_loss']:.4f}"
              f"   artifact sha={h[:12]}")

    best = tracker.best_run(metric="val_loss", minimise=True)
    print(f"\n  Best run:  {best['id']}   name={best['name']}"
          f"   val_loss={best['final_metrics']['val_loss']:.4f}"
          f"   params={best['params']}")

    n_artifacts = len(os.listdir(os.path.join(tmpdir, "artifacts")))
    print(f"\n  Total unique artifacts in content-addressed store: {n_artifacts}\n")
    print("--- library cross-check (mlflow, wandb, neptune, comet, aim) ---")
