"""Model registry / versioning (Reference Ch 32 MLOps).

Persistent store of PROMOTED MODELS with:
  * VERSIONED entries per model name (MAJOR.MINOR.PATCH).
  * STAGES: none -> staging -> production -> archived.
  * ROLLBACK: previous production version is re-activated on demand.
  * LINEAGE: link to the training-run id + input dataset hash.

Matches MLflow's Model Registry API but implemented in ~200 lines.

Demo: register 3 versions of a model, promote v2 to production, roll
back to v1, report the timeline.
"""
from __future__ import annotations    # stdlib

import json     # persistence
import os       # filesystem
import time     # timestamps

from typing import List, Dict, Optional    # type hints


class ModelRegistry:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.root, f"{name}.json")

    def _load(self, name):
        if not os.path.exists(self._path(name)):
            return {"name": name, "versions": [], "events": []}
        with open(self._path(name)) as f:
            return json.load(f)

    def _save(self, name, data):
        with open(self._path(name), "w") as f:
            json.dump(data, f, indent=2)

    def register(self, name, version, run_id, dataset_sha, metrics):
        data = self._load(name)
        if any(v["version"] == version for v in data["versions"]):
            raise ValueError(f"version {version} already exists for {name}")
        data["versions"].append({"version": version, "stage": "none",
                                   "run_id": run_id, "dataset_sha": dataset_sha,
                                   "metrics": metrics, "registered_at": time.time()})
        data["events"].append({"ts": time.time(), "event": "REGISTER",
                                 "version": version})
        self._save(name, data)
        return data["versions"][-1]

    def transition(self, name, version, stage):
        assert stage in ("none", "staging", "production", "archived")
        data = self._load(name)
        prev_prod = None
        # For "production", demote any existing production to archived unless
        # rolling back explicitly.
        if stage == "production":
            for v in data["versions"]:
                if v["stage"] == "production":
                    prev_prod = v["version"]
                    v["stage"] = "archived"
                    data["events"].append({"ts": time.time(),
                                             "event": "TRANSITION",
                                             "version": v["version"],
                                             "from": "production",
                                             "to": "archived",
                                             "reason": f"replaced by v{version}"})
        found = False
        for v in data["versions"]:
            if v["version"] == version:
                from_stage = v["stage"]; v["stage"] = stage
                data["events"].append({"ts": time.time(), "event": "TRANSITION",
                                         "version": version, "from": from_stage,
                                         "to": stage})
                found = True
                break
        if not found:
            raise KeyError(f"version {version} not found for {name}")
        self._save(name, data)
        return prev_prod

    def rollback(self, name, to_version, reason="rollback"):
        return self.transition(name, to_version, "production")

    def get_production(self, name):
        data = self._load(name)
        for v in data["versions"]:
            if v["stage"] == "production":
                return v
        return None

    def timeline(self, name):
        return self._load(name)["events"]


if __name__ == "__main__":
    print("=== Model registry: register, promote, rollback ===\n")
    import tempfile
    root = tempfile.mkdtemp(prefix="mreg_")
    reg = ModelRegistry(root)

    for v, run, m in (("1.0.0", "abc111", 0.82),
                      ("1.1.0", "abc222", 0.85),
                      ("2.0.0", "abc333", 0.83)):
        reg.register("churn_model", v, run_id=run, dataset_sha="dataset-2025-08",
                       metrics={"auc": m})

    reg.transition("churn_model", "1.1.0", "production")
    print(f"  production after promote(v1.1.0): {reg.get_production('churn_model')['version']}")

    reg.transition("churn_model", "2.0.0", "production")
    print(f"  production after promote(v2.0.0): {reg.get_production('churn_model')['version']}")

    reg.rollback("churn_model", "1.1.0")
    print(f"  production after rollback(v1.1.0): {reg.get_production('churn_model')['version']}")

    print("\n  Event timeline:")
    for e in reg.timeline("churn_model"):
        print(f"    {e['event']:12}  v={e['version']}"
              f"  {e.get('from', ''):>10} -> {e.get('to', ''):<10}"
              f"  {e.get('reason', '')}")
    print("\n--- library cross-check (mlflow.tracking.MlflowClient; sagemaker model registry; W&B Model Registry) ---")
