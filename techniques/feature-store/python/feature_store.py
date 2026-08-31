"""Feature store (Reference Ch 32 MLOps).

A FEATURE STORE serves the SAME feature values to
  * TRAINING   (batch, historical, point-in-time correct)
  * SERVING    (real-time, low-latency, current values)

Training/serving SKEW is the classic ML production bug: a feature is
computed slightly differently online vs offline, and the model's
production accuracy silently degrades.

MINIMAL feature store abstractions (Feast / Tecton / Vertex AI):
  * ENTITY   (e.g. user_id, transaction_id).
  * FEATURE VIEW  = named set of columns keyed by entity.
  * OFFLINE STORE (batch DB, parquet, warehouse).
  * ONLINE STORE  (KV store like Redis / DynamoDB).
  * MATERIALIZATION  = sync offline -> online, respecting event_time.
  * POINT-IN-TIME JOIN = fetch feature value as of a training label
    timestamp (never future-leaks).

Here we implement a tiny in-memory feature store with:
  * offline_get_features(entity_ids, event_ts)  -> point-in-time correct
  * online_get_features(entity_ids)             -> latest values
  * materialize()                                -> refresh online store
And a DEMO showing:
  * point-in-time join returns HISTORICAL values (no leakage);
  * skew detection when online / offline computations diverge.
"""
from __future__ import annotations    # stdlib

from collections import defaultdict   # dict of lists
from typing import Dict, List, Tuple  # type hints

import numpy as np    # numerical arrays


class FeatureStore:
    def __init__(self):
        # Offline: entity_id -> list of (event_time, feature_dict).
        self.offline: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
        # Online: entity_id -> latest feature_dict.
        self.online: Dict[int, dict] = {}

    def write_offline(self, entity_id, event_time, features):
        self.offline[entity_id].append((event_time, dict(features)))
        self.offline[entity_id].sort(key=lambda x: x[0])

    def materialize(self):
        """Sync latest offline value into online store."""
        for eid, rows in self.offline.items():
            latest_t, latest_feats = max(rows, key=lambda x: x[0])
            self.online[eid] = dict(latest_feats)

    def offline_get_features(self, entity_ids, event_ts):
        """Point-in-time correct: return the latest feature values with
        event_time <= event_ts, per (entity_id, request_ts) pair."""
        out = []
        for eid, ts in zip(entity_ids, event_ts):
            rows = [row for row in self.offline.get(eid, []) if row[0] <= ts]
            out.append(rows[-1][1] if rows else {})
        return out

    def online_get_features(self, entity_ids):
        return [self.online.get(eid, {}) for eid in entity_ids]


def train_skew(offline_feats, online_feats, tolerance=1e-6):
    """Compare feature-by-feature between offline and online snapshots."""
    diffs = []
    for k in offline_feats:
        if k not in online_feats:
            diffs.append((k, offline_feats[k], None, "missing_online"))
            continue
        d = abs(offline_feats[k] - online_feats[k])
        if d > tolerance:
            diffs.append((k, offline_feats[k], online_feats[k], d))
    return diffs


if __name__ == "__main__":
    print("=== Feature store: point-in-time joins + train/serve skew detection ===\n")
    store = FeatureStore()

    # Write historical values for user 42 and user 99.
    store.write_offline(42, event_time=100, features={"amount_mean_30d": 20.0, "n_txn_30d": 3})
    store.write_offline(42, event_time=200, features={"amount_mean_30d": 25.0, "n_txn_30d": 5})
    store.write_offline(42, event_time=300, features={"amount_mean_30d": 30.0, "n_txn_30d": 8})
    store.write_offline(99, event_time=150, features={"amount_mean_30d": 5.0, "n_txn_30d": 1})
    store.write_offline(99, event_time=250, features={"amount_mean_30d": 12.0, "n_txn_30d": 4})

    # POINT-IN-TIME JOIN: build training data as of ts=250 for user 42 and ts=200 for user 99.
    print("  Point-in-time offline join:")
    feats = store.offline_get_features([42, 99], [250, 200])
    for eid, ts, f in zip([42, 99], [250, 200], feats):
        print(f"    user={eid}  event_ts={ts}   features={f}")
    print("  (No future values used; e.g. user 42 at ts=250 sees value from ts=200, not from ts=300.)\n")

    # MATERIALIZE + ONLINE FETCH.
    store.materialize()
    print("  Online fetch after materialize (latest):")
    for eid, f in zip([42, 99], store.online_get_features([42, 99])):
        print(f"    user={eid}   features={f}")

    # TRAIN/SERVE SKEW: simulate an online transformation bug.
    # The training pipeline logged amount_mean_30d in dollars; the serving
    # pipeline accidentally reports amount_mean_30d in CENTS.
    offline_feats = store.offline_get_features([42], [300])[0]
    online_buggy = dict(store.online[42])
    online_buggy["amount_mean_30d"] = online_buggy["amount_mean_30d"] * 100
    diffs = train_skew(offline_feats, online_buggy, tolerance=1e-6)
    print("\n  Train/serve SKEW detected:")
    for name, off_v, on_v, d in diffs:
        print(f"    feature={name}   offline={off_v}   online={on_v}   diff={d}")

    print("\n--- library cross-check (feast, tecton, sagemaker-feature-store, vertex-ai-feature-store) ---")
