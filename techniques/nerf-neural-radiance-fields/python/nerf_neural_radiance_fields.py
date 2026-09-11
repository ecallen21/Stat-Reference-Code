"""NeRF - Neural Radiance Fields (Reference Sec 47.225).

Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi & Ng 2020
'NeRF: Representing Scenes as Neural Radiance Fields for View
Synthesis', ECCV. Represents a scene as an MLP:

    F_theta : (x, y, z, theta_view, phi_view) -> (RGB, sigma)

Rendering a pixel: cast a ray, sample N points along it, query
the MLP, and volume-composite:

    C(r) = sum_i T_i * (1 - exp(-sigma_i delta_i)) * c_i
    T_i = exp(-sum_{j<i} sigma_j delta_j)

Trained from multi-view photos with a pixel MSE loss. Slow (hours
per scene) — largely superseded by Gaussian Splatting for real-
time work.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_scene_density(xyz):
    """Toy density: high inside a sphere of radius 2, low outside."""
    r = np.linalg.norm(xyz, axis=-1)
    return np.exp(-(r - 2) ** 2 * 3) * 5


def toy_scene_color(xyz):
    """Toy color: red at +x, blue at -x, green at +y."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([np.clip(0.5 + 0.3 * x, 0, 1),
                       np.clip(0.5 + 0.3 * y, 0, 1),
                       np.clip(0.5 - 0.3 * x, 0, 1)], axis=-1)


def volume_render(ray_o, ray_d, near=1.0, far=5.0, n_samples=64):
    """Composite color along a single ray via numerical volume rendering."""
    t = np.linspace(near, far, n_samples)
    delta = t[1] - t[0]
    xyz = ray_o + t[:, None] * ray_d[None, :]
    sigma = toy_scene_density(xyz)                               # (n_samples,)
    color = toy_scene_color(xyz)                                 # (n_samples, 3)
    # Transmittance
    alpha = 1 - np.exp(-sigma * delta)
    T = np.cumprod(np.concatenate([[1.0], 1 - alpha[:-1]]))
    weights = T * alpha
    C = (weights[:, None] * color).sum(axis=0)
    depth = float((weights * t).sum())
    return C, depth


if __name__ == "__main__":
    print("=== NeRF (Mildenhall et al 2020 ECCV) ===\n")

    # Render a 32x32 image from a camera at (0, 0, -5) looking toward +z
    H = W = 32
    focal = 24.0
    origin = np.array([0.0, 0.0, -5.0])
    image = np.zeros((H, W, 3))
    depth = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            # Ray direction through pixel (i, j)
            d = np.array([(j - W / 2) / focal, -(i - H / 2) / focal, 1.0])
            d = d / np.linalg.norm(d)
            C, dep = volume_render(origin, d, near=3.0, far=7.0, n_samples=64)
            image[i, j] = C
            depth[i, j] = dep

    print(f"  Rendered {H}x{W} image via volume rendering along {H*W} rays x 64 samples/ray")
    print(f"  RGB range: [{image.min():.2f}, {image.max():.2f}], depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    print(f"  Fraction of pixels with nonzero coverage: {float((depth > 3.05).mean()) * 100:.1f}%")
    print(f"  Mean depth of covered pixels: {depth[depth > 3.05].mean():.2f}")

    print(f"\n  Real NeRF replaces the toy density / color with an MLP that maps")
    print(f"  (x, y, z, viewdir) -> (RGB, sigma), fit from 100+ multi-view photos.")

    print("\n--- library cross-check (nerfstudio; nvlabs/instant-ngp; bmild/nerf) ---")
