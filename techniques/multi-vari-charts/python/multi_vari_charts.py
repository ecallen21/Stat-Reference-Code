"""Multi-vari charts (Reference Sec 37.12).

Seder (1950). Visualise multiple sources of variation in the same
plot to see where the dominant variability lives BEFORE running DOE.

Three families of variation:
  * WITHIN-piece (positional): different measurements on the same unit.
  * BETWEEN-piece (piece-to-piece): different units in the same time
    period.
  * TEMPORAL (time-to-time): shift-to-shift, day-to-day.

The multi-vari chart stacks nested boxplots / ranges so the eye picks
out which layer dominates.  Numerical summary: analyse the range of
means at each level.

Here we implement a compact variance-source decomposition that mirrors
a multi-vari analysis on synthetic 3-level data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def variance_decomposition(values, time, piece):
    """Return within-piece + between-piece + between-time variance components."""
    within = []
    between_piece = []
    time_means = []
    for t in np.unique(time):
        mt = time == t
        piece_means = []
        for p in np.unique(piece[mt]):
            mp = mt & (piece == p)
            vals = values[mp]
            within.append(vals.var(ddof=1) if len(vals) > 1 else 0.0)
            piece_means.append(vals.mean())
        between_piece.append(np.var(piece_means, ddof=1) if len(piece_means) > 1 else 0.0)
        time_means.append(np.mean(piece_means))
    return {
        "within_mean_var": float(np.mean(within)),
        "between_piece_mean_var": float(np.mean(between_piece)),
        "between_time_var": float(np.var(time_means, ddof=1)) if len(time_means) > 1 else 0.0,
    }


if __name__ == "__main__":
    print("=== Multi-vari variation-source decomposition ===\n")
    rng = np.random.default_rng(0)
    n_times = 5; n_pieces = 4; n_measures = 3
    values = []; time = []; piece = []
    time_effect = rng.normal(0, 0.4, n_times)         # temporal variation
    for t in range(n_times):
        for p in range(n_pieces):
            piece_effect = rng.normal(0, 0.3)         # piece-to-piece
            for k in range(n_measures):
                values.append(10 + time_effect[t] + piece_effect + rng.normal(0, 0.15))
                time.append(t); piece.append(p)
    values = np.array(values); time = np.array(time); piece = np.array(piece)

    r = variance_decomposition(values, time, piece)
    total = r["within_mean_var"] + r["between_piece_mean_var"] + r["between_time_var"]
    print(f"  Variance sources (contribution to total):")
    for k, v in r.items():
        pct = 100 * v / max(total, 1e-9)
        print(f"    {k:>30}: {v:.4f}   ({pct:.1f}%)")
    print(f"    total                        : {total:.4f}")

    print("\n  Multi-vari analysis picks the largest layer as the target for improvement.\n")
    print("--- library cross-check (R SixSigma::ss.ci; Python custom) ---")
