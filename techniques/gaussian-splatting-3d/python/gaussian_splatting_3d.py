"""3D Gaussian Splatting (Reference Sec 47.224).

Kerbl, Kopanas, Leimkuehler & Drettakis 2023 '3D Gaussian
Splatting for Real-Time Radiance Field Rendering', SIGGRAPH.
Represents a scene as MILLIONS of 3D GAUSSIANS with:

    position mu_i in R^3, covariance Sigma_i (3x3 anisotropic),
    color c_i (spherical harmonics), opacity alpha_i in [0, 1].

Render: project each Gaussian to 2D (splat), depth-sort,
alpha-composite. Trained end-to-end from multi-view photos via
differentiable rasterisation.

Real-time (100+ fps) on modern GPUs at NeRF-quality; much faster
than NeRF's volumetric ray-marching.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def project_gaussian_to_2d(mu, sigma, focal, image_size):
    """Project a 3D Gaussian center + covariance to a 2D screen-space Gaussian."""
    # Simple perspective projection ignoring z-depth camera transform
    z = max(mu[2], 0.1)
    x_2d = focal * mu[0] / z + image_size / 2
    y_2d = focal * mu[1] / z + image_size / 2
    # Projected covariance (Jacobian scaling; simplified)
    sigma_2d = sigma[:2, :2] * (focal / z) ** 2
    return np.array([x_2d, y_2d]), sigma_2d


def render_splats(gaussians, image_size=64, focal=50.0):
    """Depth-sort + alpha-composite Gaussians onto a 64x64 canvas."""
    canvas = np.zeros((image_size, image_size, 3))
    alpha_accum = np.zeros((image_size, image_size))
    # Sort by depth (front to back)
    sorted_g = sorted(gaussians, key=lambda g: g["mu"][2])
    grid_x, grid_y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    grid = np.stack([grid_x, grid_y], axis=-1).astype(float)
    for g in sorted_g:
        mu2, sigma2 = project_gaussian_to_2d(g["mu"], g["sigma"], focal, image_size)
        # Gaussian falloff (isotropic-approx)
        det = max(np.linalg.det(sigma2), 1e-6)
        inv = np.linalg.inv(sigma2 + 1e-3 * np.eye(2))
        diff = grid - mu2
        exponent = -0.5 * np.einsum("...i,ij,...j", diff, inv, diff)
        gaussian_2d = np.exp(exponent) / np.sqrt(det + 1e-6)
        gaussian_2d /= gaussian_2d.max() + 1e-6
        # Alpha-composite (front-to-back)
        a = g["alpha"] * gaussian_2d
        for c in range(3):
            canvas[:, :, c] += (1 - alpha_accum) * a * g["color"][c]
        alpha_accum += (1 - alpha_accum) * a
    return canvas, alpha_accum


if __name__ == "__main__":
    print("=== 3D Gaussian Splatting (Kerbl et al 2023 SIGGRAPH) ===\n")
    rng = np.random.default_rng(0)

    # Create a small scene of 20 Gaussians at various depths
    n_gaussians = 20
    gaussians = []
    for i in range(n_gaussians):
        mu = rng.uniform(-5, 5, size=3)
        mu[2] = rng.uniform(1, 10)                                # positive depth
        eig = rng.uniform(0.1, 0.5, size=3)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        sigma = Q @ np.diag(eig ** 2) @ Q.T
        gaussians.append({"mu": mu, "sigma": sigma,
                             "color": rng.uniform(0, 1, size=3),
                             "alpha": rng.uniform(0.2, 0.8)})

    canvas, alpha = render_splats(gaussians, image_size=48, focal=30.0)
    print(f"  Scene: {n_gaussians} 3D Gaussians (position + covariance + color + alpha)")
    print(f"  Rendered image: {canvas.shape}, alpha coverage: "
          f"{float((alpha > 0.05).mean()) * 100:.1f}% of pixels")
    print(f"  RGB range: [{canvas.min():.2f}, {canvas.max():.2f}]")

    print(f"\n  Real 3DGS scenes use 500 k - 5 M Gaussians and render at 100+ fps on GPU.")
    print(f"  Optimisation from 100+ multi-view photos via differentiable rasterisation.")

    print("\n--- library cross-check (graphdeco-inria/gaussian-splatting; splatfacto in Nerfstudio) ---")
