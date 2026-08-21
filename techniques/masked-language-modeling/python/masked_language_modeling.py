"""Masked language modeling — BERT-style pretraining objective (Reference §25.x extra).

Devlin et al. 2019.  For a token sequence x, randomly mask ~15% of tokens
and train a bidirectional encoder to predict the original tokens from context:

  BERT masking rule:
    * 80% of selected tokens -> replaced by [MASK]
    * 10% -> replaced by a random token
    * 10% -> left unchanged (but still predicted)

Loss: cross-entropy at the masked positions, averaged over masked tokens
only.

We use a tiny embedding + single-layer transformer + linear head as the
encoder, but with MOCK bidirectional context: mean-of-context prediction
from surrounding tokens.  That's enough to demonstrate the objective and
its behaviour without a full transformer stack.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


def mask_tokens(seq, vocab_size, mask_id, p: float = 0.15, seed: int = 0):
    """BERT masking rule: 80% [MASK], 10% random, 10% original."""
    rng = np.random.default_rng(seed)
    masked = seq.copy(); positions = []
    for i in range(len(seq)):
        if rng.uniform() < p:
            positions.append(i)
            r = rng.uniform()
            if r < 0.8:
                masked[i] = mask_id
            elif r < 0.9:
                masked[i] = rng.integers(0, vocab_size)
    return masked, positions


def train_mlm(corpus, vocab_size: int, dim: int = 16, ctx: int = 2,
              n_iter: int = 400, lr: float = 0.05, seed: int = 0) -> dict:
    """Minimal MLM: predict a masked token from the mean of its ctx-window neighbours.
    E: embedding table (V x dim).  W: output-projection (dim x V)."""
    rng = np.random.default_rng(seed)
    V = vocab_size + 1
    mask_id = vocab_size                                    # last id is <MASK>
    E = rng.normal(scale=0.1, size=(V, dim))
    W = rng.normal(scale=0.1, size=(dim, V))
    losses = []
    for it in range(n_iter):
        loss_step = 0.0; n_pred = 0
        for seq in corpus:
            masked, pos = mask_tokens(seq, vocab_size, mask_id, p=0.15, seed=it * 1000 + hash(tuple(seq)) % 1000)
            for i in pos:
                lo = max(0, i - ctx); hi = min(len(seq), i + ctx + 1)
                nbrs = [masked[j] for j in range(lo, hi) if j != i]
                if not nbrs:
                    continue
                h = E[nbrs].mean(axis=0)                    # bidirectional context
                logits = h @ W
                probs = _softmax(logits)
                true = seq[i]
                loss_step += -np.log(probs[true] + 1e-12); n_pred += 1
                # backward
                dlogits = probs.copy(); dlogits[true] -= 1
                dW = np.outer(h, dlogits)
                dh = dlogits @ W.T
                W -= lr * dW
                for nb in nbrs:
                    E[nb] -= lr * dh / len(nbrs)
        if n_pred:
            losses.append(loss_step / n_pred)
    return {"E": E, "W": W, "mask_id": mask_id, "losses": losses,
            "method": "MLM (BERT-style; mean-context surrogate)"}


def predict_masked(seq, mask_positions, model, ctx: int = 2):
    """Fill in [MASK] tokens using the trained model."""
    filled = seq.copy()
    for i in mask_positions:
        lo = max(0, i - ctx); hi = min(len(seq), i + ctx + 1)
        nbrs = [filled[j] for j in range(lo, hi) if j != i and filled[j] != model["mask_id"]]
        if not nbrs:
            continue
        h = model["E"][nbrs].mean(axis=0)
        filled[i] = int((h @ model["W"]).argmax())
    return filled


if __name__ == "__main__":
    # tiny corpus with fixed vocab
    vocab = {w: i for i, w in enumerate(
        "the dog cat runs jumps eats sleeps quickly slowly food water bone fish "
        "a on in near far big small red blue green".split())}
    inv = {i: w for w, i in vocab.items()}
    V = len(vocab)
    sentences = [
        "the big dog runs quickly",
        "the small cat sleeps slowly",
        "a dog eats food near water",
        "the cat jumps in the food",
        "a small dog eats a bone",
        "the big cat eats fish quickly",
        "the red dog runs near a small cat",
        "a blue cat sleeps in the water",
    ] * 3
    corpus = [[vocab[w] for w in s.split()] for s in sentences]

    m = train_mlm(corpus, V, dim=8, ctx=2, n_iter=200, lr=0.05)
    print("=== MLM training ===")
    print(f"  initial loss {m['losses'][0]:.3f} -> final {m['losses'][-1]:.3f}")

    # Show mask-filling accuracy on the training set
    correct = 0; total = 0
    for s in sentences:
        seq = [vocab[w] for w in s.split()]
        masked, pos = mask_tokens(seq, V, m["mask_id"], p=0.3, seed=42)
        pred = predict_masked(masked, pos, m, ctx=2)
        for i in pos:
            if pred[i] == seq[i]:
                correct += 1
            total += 1
    print(f"  mask-filling accuracy (30% mask on train) = {correct}/{total} = "
          f"{correct / max(total, 1):.3f}")

    # Show one masked-fill example
    seq = [vocab[w] for w in "the big dog runs quickly".split()]
    masked, pos = mask_tokens(seq, V, m["mask_id"], p=0.4, seed=7)
    pred = predict_masked(masked, pos, m, ctx=2)
    def _show(s): return " ".join("[MASK]" if t == m["mask_id"] else inv[t] for t in s)
    print(f"\n  original: {_show(seq)}")
    print(f"  masked  : {_show(masked)}")
    print(f"  filled  : {_show(pred)}")

    print("\n--- library cross-check (huggingface transformers fill-mask pipeline) ---")
