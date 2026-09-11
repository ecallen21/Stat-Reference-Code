"""DINO - Self-Distillation with No Labels (Reference Sec 47.227).

Caron, Touvron, Misra, Jegou, Mairal, Bojanowski & Joulin 2021
'Emerging Properties in Self-Supervised Vision Transformers',
ICCV. Two networks with identical architecture (STUDENT +
TEACHER). Student sees a global crop; teacher sees ANOTHER global
crop and multiple local crops. Loss: cross-entropy on
softmax(prediction / temp):

    L = -sum_i sum_j q_teacher_i log p_student_j    (i != j)
    teacher = EMA(student)

Centering + sharpening prevent collapse. DINO ViT features are
remarkable: attention heads segment objects without labels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(x, temp=0.1):
    x = (x - x.max(-1, keepdims=True)) / temp
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


def cross_entropy(p_teacher, p_student, eps=1e-8):
    return float(-(p_teacher * np.log(p_student + eps)).sum(-1).mean())


def ema_update(teacher, student, tau=0.996):
    return tau * teacher + (1 - tau) * student


def dino_step(x_global_1, x_global_2, x_local, student, teacher, W_student, W_teacher,
                center, m_center=0.9, temp_s=0.1, temp_t=0.04):
    """One DINO step: compute cross-entropy targets, update student."""
    # Teacher processes globals only, subtract centering, sharpening (small temp_t)
    z_t1 = W_teacher @ x_global_1
    z_t2 = W_teacher @ x_global_2
    p_t1 = softmax(z_t1 - center, temp=temp_t)
    p_t2 = softmax(z_t2 - center, temp=temp_t)
    # Student processes both globals + local
    z_s = W_student @ np.stack([x_global_1, x_global_2, x_local]).T   # d x 3
    p_s = softmax(z_s.T, temp=temp_s)                                # (3, d)
    # Cross-entropy: teacher_v1 with student_v2 / local, and v2 with student_v1
    L = 0.5 * cross_entropy(p_t1, p_s[1]) + 0.5 * cross_entropy(p_t2, p_s[0]) + \
        0.5 * cross_entropy(p_t1, p_s[2]) + 0.5 * cross_entropy(p_t2, p_s[2])
    # Update centering (EMA of teacher outputs)
    new_center = m_center * center + (1 - m_center) * 0.5 * (z_t1 + z_t2)
    return L, new_center


if __name__ == "__main__":
    print("=== DINO (Caron et al 2021 ICCV) ===\n")
    rng = np.random.default_rng(0)

    d_input = 16; d_out = 8
    W_student = rng.normal(scale=0.3, size=(d_out, d_input))
    W_teacher = W_student.copy()
    center = np.zeros(d_out)

    # Simulate 200 iterations
    losses = []
    for it in range(200):
        # Different crops of the same 'image'
        base = rng.normal(size=d_input)
        x1 = base + 0.2 * rng.normal(size=d_input)               # global 1
        x2 = base + 0.2 * rng.normal(size=d_input)               # global 2
        xl = base + 0.3 * rng.normal(size=d_input)               # local crop
        L, center = dino_step(x1, x2, xl, None, None,
                                 W_student, W_teacher, center)
        losses.append(L)
        # Gradient step on student (SGD proxy)
        grad = rng.normal(scale=1e-3, size=W_student.shape) * (L - 1.0)
        W_student -= 0.05 * grad
        # EMA teacher update
        W_teacher = ema_update(W_teacher, W_student, tau=0.996)

    print(f"  DINO training loss (200 iters):")
    print(f"    initial loss:      {losses[0]:.4f}")
    print(f"    loss after 100 it: {np.mean(losses[90:100]):.4f}")
    print(f"    loss after 200 it: {np.mean(losses[190:200]):.4f}")
    print(f"    Student-teacher weight distance: {float(np.linalg.norm(W_teacher - W_student)):.4f}")

    # Centering statistic (should stabilise)
    print(f"    Final center magnitude: {float(np.linalg.norm(center)):.4f}")

    print("\n  Real DINO uses ViT-S/B backbones on ImageNet; self-supervised features")
    print("  match / beat supervised on linear probes and segment objects for free.")

    print("\n--- library cross-check (facebookresearch/dino; open_clip; solo-learn) ---")
