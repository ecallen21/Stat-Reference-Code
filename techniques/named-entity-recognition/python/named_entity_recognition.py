"""Named-entity recognition (Reference §25.8).

Sequence tagging: for each token in a sentence, assign a tag from a small
inventory (e.g. B-PER, I-PER, B-ORG, I-ORG, O for outside).

We implement a hidden Markov model with Viterbi decoding:
    y_1, ..., y_n ~ Markov
    x_i | y_i     ~ Categorical (emission)
The forward-backward + Viterbi identities are the same as in `hmm`; here
we specialise to a small NER lexicon and BIO tag set.

Production NER uses BiLSTM-CRF (Lample 2016) or transformer + linear head
(BERT + NER); the HMM here is a solid classical baseline and pedagogical
example.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def train_hmm_ner(sentences, tags) -> dict:
    """Estimate HMM parameters by counting on labelled sentences."""
    # collect vocabs
    tag_set = sorted({t for ts in tags for t in ts})
    word_set = sorted({w for ws in sentences for w in ws})
    tag_ix = {t: i for i, t in enumerate(tag_set)}
    word_ix = {w: i for i, w in enumerate(word_set)}
    K = len(tag_set); V = len(word_set)
    # counts
    init = np.zeros(K); trans = np.zeros((K, K)); emit = np.zeros((K, V))
    for ws, ts in zip(sentences, tags):
        init[tag_ix[ts[0]]] += 1
        for i, (w, t) in enumerate(zip(ws, ts)):
            emit[tag_ix[t], word_ix[w]] += 1
            if i + 1 < len(ts):
                trans[tag_ix[t], tag_ix[ts[i + 1]]] += 1
    # normalise + Laplace smoothing
    init = (init + 1) / (init.sum() + K)
    trans = (trans + 1) / (trans.sum(axis=1, keepdims=True) + K)
    emit = (emit + 1) / (emit.sum(axis=1, keepdims=True) + V)
    return {"tag_set": tag_set, "word_set": word_set,
            "tag_ix": tag_ix, "word_ix": word_ix,
            "init": init, "trans": trans, "emit": emit}


def viterbi_decode(sentence, model) -> list:
    """Decode most-likely tag sequence for a new sentence."""
    T = len(sentence); K = len(model["tag_set"])
    log_init = np.log(model["init"])
    log_trans = np.log(model["trans"])
    log_emit = np.log(model["emit"])
    unk_emit = np.log(1.0 / len(model["word_set"]))       # uniform for OOV
    delta = np.zeros((T, K)); psi = np.zeros((T, K), dtype=int)
    def _emit_col(w):
        i = model["word_ix"].get(w, None)
        return log_emit[:, i] if i is not None else np.full(K, unk_emit)
    delta[0] = log_init + _emit_col(sentence[0])
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_trans        # (K_prev, K_now)
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + _emit_col(sentence[t])
    # backtrack
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return [model["tag_set"][i] for i in path]


def _f1_by_entity(y_true, y_pred, tag_set):
    """Simple entity-level F1 over BIO tags."""
    def _spans(seq):
        spans = []; i = 0
        while i < len(seq):
            if seq[i].startswith("B-"):
                etype = seq[i][2:]; j = i + 1
                while j < len(seq) and seq[j] == f"I-{etype}":
                    j += 1
                spans.append((etype, i, j))
                i = j
            else:
                i += 1
        return spans
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        st = set(_spans(yt)); sp = set(_spans(yp))
        tp += len(st & sp); fp += len(sp - st); fn += len(st - sp)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return {"precision": p, "recall": r, "F1": f,
            "TP": tp, "FP": fp, "FN": fn}


if __name__ == "__main__":
    # tiny corpus with BIO tagging for PER (person) and ORG (organisation)
    train_sents = [
        ("Alice went to Acme Corp on Monday .".split(),
         "B-PER O   O  B-ORG I-ORG O  O      O".split()),
        ("Bob visited Globex Industries yesterday .".split(),
         "B-PER O       B-ORG  I-ORG        O          O".split()),
        ("Carol works for Initech .".split(),
         "B-PER O     O   B-ORG   O".split()),
        ("David and Emma joined Umbrella Corp last week .".split(),
         "B-PER O   B-PER O      B-ORG    I-ORG O    O    O".split()),
    ]
    sents = [s for s, _ in train_sents]
    tags = [t for _, t in train_sents]
    model = train_hmm_ner(sents, tags)

    test_sents = [
        "Alice joined Globex Industries yesterday .".split(),
        "Bob works for Acme Corp .".split(),
        "David and Carol founded Initech last week .".split(),
    ]
    print("=== HMM NER (Viterbi decoding) ===")
    for s in test_sents:
        pred = viterbi_decode(s, model)
        print(f"\n  tokens: {s}")
        print(f"  tags  : {pred}")

    # evaluate on the training set (sanity)
    predicted = [viterbi_decode(s, model) for s in sents]
    r = _f1_by_entity(tags, predicted, model["tag_set"])
    print(f"\n  entity-level TP={r['TP']} FP={r['FP']} FN={r['FN']} "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['F1']:.3f}   (training set)")

    print("\n--- library cross-check (spacy nlp.pipe; nltk.ne_chunk; flair; transformers pipeline) ---")
