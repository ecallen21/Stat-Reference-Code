"""Learning-rate schedules (Reference §27.x extra).

Five workhorse LR schedules:

  * constant:        lr(t) = lr_0
  * step decay:      lr(t) = lr_0 * gamma^floor(t / step)
  * cosine:          lr(t) = lr_min + (lr_0 - lr_min) * (1 + cos(pi t / T)) / 2
  * linear warmup + cosine decay: standard transformer / LLM recipe
  * one-cycle (Smith 2018): linear up to lr_max in first half, then cosine down
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def constant(lr_0: float, T: int) -> np.ndarray:
    return np.full(T, lr_0)


def step_decay(lr_0: float, T: int, step: int = 30, gamma: float = 0.1) -> np.ndarray:
    return np.array([lr_0 * gamma ** (t // step) for t in range(T)])


def cosine(lr_0: float, T: int, lr_min: float = 0.0) -> np.ndarray:
    return np.array([lr_min + (lr_0 - lr_min) * (1 + math.cos(math.pi * t / T)) / 2
                     for t in range(T)])


def warmup_cosine(lr_0: float, T: int, warmup: int = 10, lr_min: float = 0.0) -> np.ndarray:
    out = np.zeros(T)
    for t in range(T):
        if t < warmup:
            out[t] = lr_0 * (t + 1) / warmup
        else:
            frac = (t - warmup) / max(T - warmup, 1)
            out[t] = lr_min + (lr_0 - lr_min) * (1 + math.cos(math.pi * frac)) / 2
    return out


def one_cycle(lr_max: float, T: int, lr_init: float = None,
              lr_final: float = None) -> np.ndarray:
    if lr_init is None: lr_init = lr_max / 25
    if lr_final is None: lr_final = lr_max / 1000
    half = T // 2
    out = np.zeros(T)
    for t in range(T):
        if t < half:
            out[t] = lr_init + (lr_max - lr_init) * (t / max(half, 1))
        else:
            frac = (t - half) / max(T - half, 1)
            out[t] = lr_final + (lr_max - lr_final) * (1 + math.cos(math.pi * frac)) / 2
    return out


if __name__ == "__main__":
    T = 100; lr_0 = 0.1
    schedules = {
        "constant":         constant(lr_0, T),
        "step (30, 0.1)":   step_decay(lr_0, T, step=30, gamma=0.1),
        "cosine":           cosine(lr_0, T),
        "warmup+cosine":    warmup_cosine(lr_0, T, warmup=10),
        "one-cycle":        one_cycle(lr_0, T),
    }
    print(f"=== LR schedules over T={T} steps, lr_0 = {lr_0} ===")
    print(f"  {'schedule':>16}    lr(t=0)  lr(t=10)  lr(t=50)  lr(t=99)")
    for name, sched in schedules.items():
        print(f"  {name:>16}     {sched[0]:>7.4f}  {sched[10]:>7.4f}  "
              f"{sched[50]:>7.4f}  {sched[-1]:>7.4f}")

    # sanity: warmup+cosine starts near 0, peaks at warmup, ends near 0
    ws = schedules["warmup+cosine"]
    print(f"\n  warmup+cosine peak at step {int(np.argmax(ws))} of {T - 1}, "
          f"peak lr = {ws.max():.4f}")

    # one-cycle
    oc = schedules["one-cycle"]
    print(f"  one-cycle peak at step {int(np.argmax(oc))} of {T - 1}, "
          f"peak lr = {oc.max():.4f}")

    print("\n--- library cross-check (torch.optim.lr_scheduler / keras.callbacks) ---")
