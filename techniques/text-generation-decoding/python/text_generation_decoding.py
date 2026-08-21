"""Decoding strategies for autoregressive text generation (Reference §25.x extra).

Given a language model p(x_{t+1} | x_{1:t}), choose the next token from a
distribution shaped by:

  * greedy:            argmax p(x_{t+1} | x_{1:t})
  * beam search:       maintain top-B partial hypotheses by sum-log-prob
  * temperature:       p(x)^{1/T} then renormalise (T < 1 sharpens; T > 1 flattens)
  * top-k:             restrict sampling to the k highest-probability tokens
  * nucleus (top-p):   restrict to the smallest set whose cumulative mass >= p

Greedy / beam maximise likelihood -> repetitive; top-p / top-k inject
diversity -> more human-like.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, temperature: float = 1.0):
    z = z / max(temperature, 1e-9)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


def sample_greedy(logits) -> int:
    return int(np.argmax(logits))


def sample_temperature(logits, T: float, rng) -> int:
    p = _softmax(np.asarray(logits, dtype=float), temperature=T)
    return int(rng.choice(len(p), p=p))


def sample_topk(logits, k: int, T: float, rng) -> int:
    p = _softmax(np.asarray(logits, dtype=float), temperature=T)
    top = np.argpartition(-p, k)[:k]
    q = np.zeros_like(p); q[top] = p[top]; q = q / q.sum()
    return int(rng.choice(len(q), p=q))


def sample_topp(logits, p_thresh: float, T: float, rng) -> int:
    p = _softmax(np.asarray(logits, dtype=float), temperature=T)
    order = np.argsort(-p); cum = np.cumsum(p[order])
    cutoff = int(np.searchsorted(cum, p_thresh)) + 1
    keep = order[:cutoff]
    q = np.zeros_like(p); q[keep] = p[keep]; q = q / q.sum()
    return int(rng.choice(len(q), p=q))


def beam_search(step_fn, start, beam: int = 4, max_len: int = 12) -> tuple:
    """step_fn(prefix) -> array of log-probabilities for the next token.
    Returns the highest-scoring completed sequence."""
    beams = [(start, 0.0)]                                  # (prefix, log-score)
    for _ in range(max_len):
        candidates = []
        for prefix, score in beams:
            log_p = step_fn(prefix)
            top = np.argsort(-log_p)[:beam]
            for t in top:
                candidates.append((prefix + [int(t)], score + float(log_p[t])))
        beams = sorted(candidates, key=lambda x: -x[1])[:beam]
    return beams[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    V = 12                                                  # vocab size
    # Toy LM: bigram counts on a fixed corpus, converted to log-probs
    corpus = [[0, 1, 2, 3, 4], [0, 1, 5, 3, 4], [0, 2, 5, 6, 4],
              [0, 1, 2, 3, 7], [0, 5, 6, 7, 4], [0, 2, 3, 6, 7]]
    trans = np.zeros((V, V)) + 0.5                         # Laplace
    for s in corpus:
        for i in range(len(s) - 1):
            trans[s[i], s[i + 1]] += 1
    trans = trans / trans.sum(axis=1, keepdims=True)
    logp = np.log(trans + 1e-12)

    def _step(prefix):
        return logp[prefix[-1]]

    print("=== Decoding strategies from a toy bigram LM ===")
    greedy = [0]
    for _ in range(8):
        greedy.append(sample_greedy(logp[greedy[-1]]))
    print(f"  greedy       : {greedy}")

    for T in (0.5, 1.0, 1.5):
        rng = np.random.default_rng(1)
        seq = [0]
        for _ in range(8):
            seq.append(sample_temperature(logp[seq[-1]], T, rng))
        print(f"  temp T={T:>3}   : {seq}")

    rng = np.random.default_rng(2)
    seq = [0]
    for _ in range(8):
        seq.append(sample_topk(logp[seq[-1]], k=3, T=1.0, rng=rng))
    print(f"  top-k (k=3)  : {seq}")

    rng = np.random.default_rng(3)
    seq = [0]
    for _ in range(8):
        seq.append(sample_topp(logp[seq[-1]], p_thresh=0.9, T=1.0, rng=rng))
    print(f"  top-p (p=0.9): {seq}")

    best = beam_search(_step, [0], beam=4, max_len=8)
    print(f"  beam (B=4)   : {best[0]}   log-score = {best[1]:.3f}")

    print("\n--- library cross-check (huggingface transformers generate() with kwargs) ---")
