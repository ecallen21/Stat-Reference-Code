"""Dynamic Mode Decomposition (DMD) (Reference Sec 47.126).

Schmid 2010 'Dynamic mode decomposition of numerical and
experimental data', J Fluid Mech 656. Data-driven approximation of
the Koopman operator: given snapshots

    X = [x_0, x_1, ..., x_{n-1}],   X' = [x_1, x_2, ..., x_n]

find A minimising ||X' - A X||_F under low-rank truncation.
Exact-DMD (Tu et al 2014):

    U, S, Vt = SVD(X, rank r)
    A_tilde = U^H X' V S^{-1}
    Phi = X' V S^{-1} W     (eigenvectors of A_tilde: W, Lambda)
    x(t) = Phi exp(Omega t) b,   Omega = log(Lambda)/dt.

Predicts future snapshots + extracts spatiotemporal MODES.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def exact_dmd(X_snapshots, r=None):
    """Exact DMD (Tu et al 2014). Input columns are snapshots."""
    X = X_snapshots[:, :-1]; Xp = X_snapshots[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if r is not None:
        U = U[:, :r]; S = S[:r]; Vt = Vt[:r, :]
    S_inv = np.diag(1.0 / S)
    A_tilde = U.conj().T @ Xp @ Vt.conj().T @ S_inv
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = Xp @ Vt.conj().T @ S_inv @ W
    b = np.linalg.lstsq(Phi, X_snapshots[:, 0], rcond=None)[0]
    return {"Phi": Phi, "eigvals": eigvals, "b": b}


def dmd_predict(res, dt, t_grid):
    Lambda = res["eigvals"]
    omega = np.log(Lambda + 0j) / dt
    return np.real(res["Phi"] @ np.diag(res["b"]) @ np.exp(np.outer(omega, t_grid)))


if __name__ == "__main__":
    print("=== Dynamic Mode Decomposition (Schmid 2010) ===\n")
    rng = np.random.default_rng(0)
    # Superposition of two damped travelling waves on a 1-D spatial grid
    x = np.linspace(-5, 5, 200)
    T = 128
    dt = 0.1
    t = np.arange(T) * dt
    # Two spatial modes each with a distinct temporal frequency (complex-valued)
    mode1 = (1 / np.cosh(x + 3))
    mode2 = (1 / np.cosh(x - 3))
    snapshots = (mode1[:, None] * np.exp(2j * t[None, :])
                  + mode2[:, None] * np.exp(3j * t[None, :]))

    r = 4
    res = exact_dmd(snapshots, r=r)
    freqs = np.angle(res["eigvals"]) / dt
    print(f"  DMD rank = {r}")
    print(f"  |Eigenvalues| = {np.round(np.abs(res['eigvals']), 3)}   (unit-circle => undamped)")
    # Complex signal has only positive-freq modes; extra small-|lambda| modes are numerical noise
    print(f"  Angular freqs recovered (dominant) = {np.round(np.sort(freqs[np.abs(res['eigvals']) > 0.5]), 3)}   (truth [+2, +3])")

    # Reconstruction error
    recon = dmd_predict(res, dt, t)
    rel = float(np.linalg.norm(snapshots.real - recon.real) / np.linalg.norm(snapshots.real))
    print(f"  Relative reconstruction error = {rel:.4f}")

    print("\n--- library cross-check (pydmd Python; no established R port) ---")
