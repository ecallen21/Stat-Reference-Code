"""Self-Organizing Map - SOM (Reference Sec 47.311).

Kohonen 1982 Biol Cybern. Unsupervised 2-D grid of neurons that
learns a TOPOLOGICALLY ORDERED representation of the input:

    for each x:
        BMU = argmin_i ||x - w_i||          (best matching unit)
        for each neuron j on the grid:
            w_j <- w_j + eta * h(j, BMU) * (x - w_j)

Neighbourhood function h decays with distance from BMU (Gaussian
kernel) and with training epoch. Widely used for visualisation
(U-matrix) and exploratory clustering.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class SOM:
    def __init__(self, grid, d, rng):
        self.grid = grid                                          # e.g. (8, 8)
        self.d = d
        self.W = rng.normal(0, 1, (grid[0], grid[1], d))
        self.coords = np.array([(i, j) for i in range(grid[0]) for j in range(grid[1])])

    def _bmu(self, x):
        dist = np.sum((self.W - x) ** 2, axis=-1)
        return np.unravel_index(np.argmin(dist), self.grid)

    def fit(self, X, epochs=100, eta0=0.5, sigma0=None):
        if sigma0 is None: sigma0 = max(self.grid) / 2
        for ep in range(epochs):
            eta = eta0 * np.exp(-ep / epochs)
            sigma = sigma0 * np.exp(-ep / epochs)
            for x in X:
                bmu = self._bmu(x)
                for i in range(self.grid[0]):
                    for j in range(self.grid[1]):
                        d_ij = (i - bmu[0]) ** 2 + (j - bmu[1]) ** 2
                        h = np.exp(-d_ij / (2 * sigma ** 2))
                        self.W[i, j] += eta * h * (x - self.W[i, j])

    def u_matrix(self):
        """Neighbour-distance for visualisation."""
        H, W = self.grid; u = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                nbrs = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        nbrs.append(np.linalg.norm(self.W[i, j] - self.W[ni, nj]))
                u[i, j] = np.mean(nbrs) if nbrs else 0
        return u


if __name__ == "__main__":
    print("=== Self-Organizing Map (Kohonen 1982) ===\n")
    rng = np.random.default_rng(0)

    # 3 well-separated clusters in 4-D
    centres = np.array([[0, 0, 0, 0], [5, 5, 0, 0], [0, 0, 5, 5]])
    labels = rng.integers(0, 3, 300)
    X = centres[labels] + rng.normal(0, 0.5, (300, 4))

    som = SOM((6, 6), d=4, rng=rng)
    som.fit(X, epochs=30, eta0=0.5, sigma0=3.0)

    # Assign each sample to its BMU
    assignments = np.zeros((6, 6, 3))
    for x, lab in zip(X, labels):
        i, j = som._bmu(x); assignments[i, j, lab] += 1

    print(f"  BMU class distribution on 6x6 grid (rows/cols = grid coords):")
    for i in range(6):
        row = "  ".join(f"C{int(np.argmax(assignments[i, j]))}"
                        if assignments[i, j].sum() > 0 else " ."
                        for j in range(6))
        print(f"    {row}")

    print(f"\n  U-matrix (distances between neighbouring neurons):")
    U = som.u_matrix()
    for row in U:
        print(f"    {' '.join(f'{v:.2f}' for v in row)}")

    print(f"\n  Cluster boundaries appear as high U-matrix values -")
    print(f"  the classic Kohonen visualisation of topological order.")

    print("\n--- library cross-check (minisom Python; kohonen R; scikit-som) ---")
