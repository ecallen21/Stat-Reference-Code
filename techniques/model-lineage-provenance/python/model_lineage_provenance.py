"""Model lineage / provenance (Reference Ch 32 MLOps).

Track the DAG of everything that went into a prediction:
  RAW_DATA -> FEATURE_VIEW -> TRAINING_RUN -> MODEL_VERSION -> PREDICTION

Every node carries a content HASH; every edge records the transformation.

Answers the ops questions:
  * Which prediction rows used feature-view v3.1?
  * Which model version was v3.1 trained on?
  * Which raw ETL sources fed feature-view v3.1?
  * If a data-quality issue is found at the raw source, which
    predictions must be re-scored?

Here we implement a small in-memory `LineageGraph` (nodes + edges),
build a lineage for a demo training/prediction flow, and answer
DOWNSTREAM (blast-radius) + UPSTREAM (root-cause) queries.
"""
from __future__ import annotations    # stdlib

from collections import defaultdict, deque    # DAG queries
from dataclasses import dataclass, field       # nodes
from typing import Dict, List                   # type hints

import hashlib   # SHA-256


def _hash(x): return hashlib.sha256(str(x).encode()).hexdigest()[:12]


@dataclass
class Node:
    id: str
    kind: str            # 'raw_data', 'feature_view', 'training_run', 'model_version', 'prediction_batch'
    meta: dict = field(default_factory=dict)


class LineageGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self.children: Dict[str, List[str]] = defaultdict(list)

    def add(self, id_, kind, **meta):
        self.nodes[id_] = Node(id=id_, kind=kind, meta=meta)
        return id_

    def link(self, parent, child, edge_meta=None):
        self.parents[child].append(parent)
        self.children[parent].append(child)

    def upstream(self, node_id):
        """All ancestors reachable via parent edges (root-cause)."""
        seen, q = set(), deque([node_id])
        while q:
            v = q.popleft()
            for p in self.parents[v]:
                if p not in seen:
                    seen.add(p); q.append(p)
        return list(seen)

    def downstream(self, node_id):
        """All descendants reachable via child edges (blast-radius)."""
        seen, q = set(), deque([node_id])
        while q:
            v = q.popleft()
            for c in self.children[v]:
                if c not in seen:
                    seen.add(c); q.append(c)
        return list(seen)


if __name__ == "__main__":
    print("=== Model lineage / provenance DAG ===\n")
    g = LineageGraph()

    # Raw data sources
    raw_txn = g.add("raw:transactions_2025-08", "raw_data", sha=_hash("transactions_2025-08"))
    raw_usr = g.add("raw:users_2025-08",         "raw_data", sha=_hash("users_2025-08"))

    # Feature views
    fv_txn = g.add("fv:txn_stats_v3.1", "feature_view", sha=_hash("fv_txn_3.1"))
    fv_usr = g.add("fv:user_demo_v1.0", "feature_view", sha=_hash("fv_user_1.0"))
    g.link(raw_txn, fv_txn); g.link(raw_usr, fv_usr)

    # Training run
    run = g.add("run:2025-08-31T12:00", "training_run", seed=42, code_version="b12ffec")
    g.link(fv_txn, run); g.link(fv_usr, run)

    # Model version
    mv = g.add("model:churn_v2.0.0", "model_version",
                 metrics={"auc": 0.87}, artifact_sha=_hash("model_v2_weights"))
    g.link(run, mv)

    # Prediction batches
    p1 = g.add("pred:batch-2025-08-31-A", "prediction_batch", n=10_000)
    p2 = g.add("pred:batch-2025-08-31-B", "prediction_batch", n=5_000)
    g.link(mv, p1); g.link(mv, p2)

    print("  Nodes:")
    for n in g.nodes.values():
        print(f"    [{n.kind:16}] {n.id}   meta={n.meta}")

    # ROOT-CAUSE: what fed prediction batch A?
    up = g.upstream("pred:batch-2025-08-31-A")
    print(f"\n  Upstream (ancestors) of 'pred:batch-2025-08-31-A':")
    for u in up:
        print(f"    {u}")

    # BLAST-RADIUS: if raw:transactions_2025-08 is bad, what needs re-scoring?
    down = g.downstream("raw:transactions_2025-08")
    print(f"\n  Downstream (dependents) of 'raw:transactions_2025-08':")
    for d in down:
        print(f"    {d}")

    print("\n--- library cross-check (openlineage, marquez, mlflow model lineage, dbt lineage,"
          " datahub, amundsen) ---")
