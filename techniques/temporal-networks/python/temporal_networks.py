"""Temporal networks: time-stamped edges, temporal paths, snapshots (Reference §24.x extra).

A temporal network is a stream of edges (u_k, v_k, t_k).  Time-aware
analogues of static-graph quantities:

  * Snapshot at time t     — static graph of edges with t_k <= t (or in window [t - w, t]).
  * Time-respecting path   — sequence of edges with strictly increasing times.
  * Temporal reachability  — is there a time-respecting u -> v path?
  * Temporal centrality    — path-count / distance summaries respecting time.

We implement:
  * Sliding-window snapshot builder.
  * Foremost/earliest-arrival BFS on the ordered edge stream.
  * Temporal reachability matrix.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def snapshot(events, t_end: float, t_start: float = None):
    """Static undirected adjacency of events with t in (t_start, t_end]."""
    events = np.asarray(events)
    mask = events[:, 2] <= t_end
    if t_start is not None:
        mask &= events[:, 2] > t_start
    edges = events[mask, :2].astype(int)
    n = int(events[:, :2].max() + 1)
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    return A


def earliest_arrival(events, source, n_nodes: int) -> np.ndarray:
    """Foremost / earliest-arrival time from source to every node.

    Scan events in time order; if edge (u, v, t) fires, and arrived[u] <= t
    (source is 'known' by time t via a time-respecting path), then update
    arrived[v] to min(arrived[v], t).
    """
    events = np.asarray(events, dtype=float)
    order = np.argsort(events[:, 2], kind="mergesort")
    events = events[order]
    arrived = np.full(n_nodes, np.inf)
    arrived[source] = -np.inf                               # source known at t = -inf
    for (u, v, t) in events:
        u = int(u); v = int(v)
        # symmetric contagion (undirected)
        if arrived[u] <= t and t < arrived[v]:
            arrived[v] = t
        if arrived[v] <= t and t < arrived[u]:
            arrived[u] = t
    # replace -inf sentinel at source with 0
    arrived[source] = 0.0
    return arrived


def temporal_reachability(events, n_nodes: int) -> np.ndarray:
    R = np.zeros((n_nodes, n_nodes), dtype=int)
    for s in range(n_nodes):
        a = earliest_arrival(events, s, n_nodes)
        R[s] = (a < np.inf).astype(int)
    return R


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8; T = 100
    # random contact stream: 30 edges
    events = []
    for _ in range(30):
        u, v = rng.choice(n, 2, replace=False)
        t = float(rng.uniform(0, T))
        events.append((int(u), int(v), t))
    events = np.asarray(sorted(events, key=lambda e: e[2]))

    print(f"=== Temporal network: n={n} nodes, {len(events)} contacts on [0, {T}] ===")

    A_all = snapshot(events, t_end=T)
    print(f"  aggregated (all-time) edges: {int(A_all.sum() // 2)}   density: {A_all.sum() / (n * (n - 1)):.3f}")

    for win in (25, 50, 100):
        A_win = snapshot(events, t_end=T, t_start=T - win)
        print(f"  window (T-{win}, T]: edges = {int(A_win.sum() // 2)}")

    # earliest arrival from node 0
    arrived = earliest_arrival(events, source=0, n_nodes=n)
    print(f"\n  earliest arrival from node 0:")
    for i in range(n):
        t = arrived[i]
        print(f"    -> node {i}: {t if t < np.inf else 'unreachable'}")

    # temporal reachability
    R = temporal_reachability(events, n_nodes=n)
    # static reachability for comparison: BFS on aggregated graph components
    from collections import deque
    def _reach_static(A, s):
        n_ = A.shape[0]; seen = np.zeros(n_, dtype=int); seen[s] = 1
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.where(A[u])[0]:
                if not seen[v]:
                    seen[v] = 1; q.append(v)
        return seen
    R_static = np.array([_reach_static(A_all, s) for s in range(n)])
    print(f"\n  temporal reachability sum   = {int(R.sum())} / {n * n}")
    print(f"  static (aggregated) sum     = {int(R_static.sum())} / {n * n}")
    print(f"  (temporal <= static always: time-respecting paths are a subset)")

    print("\n--- library cross-check (Python teneto / R timeordered) ---")
