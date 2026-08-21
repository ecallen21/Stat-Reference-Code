"""Part-of-speech tagging with HMM + Viterbi (Reference §25.x extra).

Classical bigram HMM POS tagger:
    y_1, ..., y_n ~ Markov chain over tag set
    x_i | y_i     ~ Categorical over vocabulary

Trained by MLE (counts + Laplace smoothing) on a tagged corpus; decoded with
Viterbi.  Modern replacement: BiLSTM-CRF or transformer + linear head; the
HMM is the pedagogical baseline and often accurate enough for morphologically
poor tagsets.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def train_hmm_pos(sentences, tags, alpha: float = 1.0) -> dict:
    tag_set = sorted({t for ts in tags for t in ts})
    word_set = sorted({w for ws in sentences for w in ws})
    tag_ix = {t: i for i, t in enumerate(tag_set)}
    word_ix = {w: i for i, w in enumerate(word_set)}
    K = len(tag_set); V = len(word_set)
    init = np.zeros(K); trans = np.zeros((K, K)); emit = np.zeros((K, V))
    for ws, ts in zip(sentences, tags):
        init[tag_ix[ts[0]]] += 1
        for i, (w, t) in enumerate(zip(ws, ts)):
            emit[tag_ix[t], word_ix[w]] += 1
            if i + 1 < len(ts):
                trans[tag_ix[t], tag_ix[ts[i + 1]]] += 1
    init = (init + alpha) / (init.sum() + alpha * K)
    trans = (trans + alpha) / (trans.sum(axis=1, keepdims=True) + alpha * K)
    emit = (emit + alpha) / (emit.sum(axis=1, keepdims=True) + alpha * V)
    return {"tag_set": tag_set, "word_set": word_set,
            "tag_ix": tag_ix, "word_ix": word_ix,
            "init": init, "trans": trans, "emit": emit}


def viterbi_pos(sentence, model) -> list:
    T = len(sentence); K = len(model["tag_set"])
    log_init = np.log(model["init"])
    log_trans = np.log(model["trans"])
    log_emit = np.log(model["emit"])
    unk = np.log(1.0 / len(model["word_set"]))
    def _e(w):
        i = model["word_ix"].get(w)
        return log_emit[:, i] if i is not None else np.full(K, unk)
    delta = np.zeros((T, K)); psi = np.zeros((T, K), dtype=int)
    delta[0] = log_init + _e(sentence[0])
    for t in range(1, T):
        s = delta[t - 1][:, None] + log_trans
        psi[t] = s.argmax(axis=0); delta[t] = s.max(axis=0) + _e(sentence[t])
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return [model["tag_set"][i] for i in path]


if __name__ == "__main__":
    # tiny hand-labelled corpus: DT (det), NN (noun), VB (verb), JJ (adj)
    train = [
        ("the cat chased the mouse".split(), "DT NN VB DT NN".split()),
        ("a dog eats fresh meat".split(), "DT NN VB JJ NN".split()),
        ("the big cat sleeps".split(), "DT JJ NN VB".split()),
        ("the mouse hides".split(), "DT NN VB".split()),
        ("a fresh dog runs".split(), "DT JJ NN VB".split()),
        ("the small dog eats meat".split(), "DT JJ NN VB NN".split()),
    ]
    sents = [s for s, _ in train]; tags = [t for _, t in train]
    m = train_hmm_pos(sents, tags)

    tests = [
        "the dog eats a fresh mouse".split(),
        "a big cat sleeps".split(),
        "the mouse chased the small cat".split(),
    ]
    print("=== HMM POS tagger (bigram + Laplace + Viterbi) ===")
    for s in tests:
        pred = viterbi_pos(s, m)
        print(f"\n  tokens: {s}")
        print(f"  tags  : {pred}")

    # token-level accuracy on the training set
    correct = 0; total = 0
    for s, t in train:
        pred = viterbi_pos(s, m)
        correct += sum(pi == ti for pi, ti in zip(pred, t))
        total += len(t)
    print(f"\n  training-set token accuracy = {correct}/{total} = {correct / total:.3f}")

    print("\n--- library cross-check (nltk.pos_tag; spacy; stanza; flair) ---")
