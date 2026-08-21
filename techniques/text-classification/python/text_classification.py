"""Text classification: multinomial Naive Bayes + TF-IDF logistic (Reference §25.6).

Multinomial NB:
    P(y = c | x)  prop  P(y = c) * prod_w P(w | c)^{x_w}
    P(w | c) = (count(w, c) + alpha) / (sum_v count(v, c) + alpha * |V|)   (Laplace smoothing)

Logistic on TF-IDF: standard binary / multinomial logistic regression on the
TF-IDF matrix.  For text, NB is very hard to beat as a baseline; regularised
logistic + character n-grams usually wins on well-labelled corpora.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def _tf_matrix(docs, vocab=None):
    if vocab is None:
        vocab = {}
        for d in docs:
            for w in d:
                if w not in vocab:
                    vocab[w] = len(vocab)
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            if w in vocab:
                X[i, vocab[w]] = c
    return X, vocab


def fit_multinomial_nb(docs, y, alpha: float = 1.0) -> dict:
    X, vocab = _tf_matrix(docs)
    y = np.asarray(y)
    classes = np.unique(y); C = len(classes); V = X.shape[1]
    log_prior = np.zeros(C)
    log_lik = np.zeros((C, V))
    for k, c in enumerate(classes):
        idx = y == c
        log_prior[k] = np.log(idx.mean())
        counts = X[idx].sum(axis=0) + alpha
        log_lik[k] = np.log(counts / counts.sum())
    return {"vocab": vocab, "classes": classes,
            "log_prior": log_prior, "log_lik": log_lik,
            "method": "multinomial Naive Bayes"}


def predict_nb(docs, model):
    X, _ = _tf_matrix(docs, model["vocab"])
    scores = X @ model["log_lik"].T + model["log_prior"]
    return model["classes"][scores.argmax(axis=1)]


def _tfidf(docs, vocab=None):
    X, vocab = _tf_matrix(docs, vocab)
    N = len(docs)
    df = (X > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    W = X * idf[None, :]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = np.where(norms > 0, W / norms, 0.0)
    return W, vocab, idf


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def fit_multiclass_logistic(X, y, l2: float = 1.0, n_iter: int = 400,
                              lr: float = 0.5) -> dict:
    classes = np.unique(y); C = len(classes)
    Y = np.zeros((len(y), C))
    for k, c in enumerate(classes):
        Y[y == c, k] = 1
    D = X.shape[1]
    W = np.zeros((D, C))
    for _ in range(n_iter):
        P = _softmax(X @ W)
        grad = X.T @ (P - Y) + l2 * W
        W -= lr * grad / len(X)
    return {"classes": classes, "W": W}


def predict_logistic(X, model):
    return model["classes"][_softmax(X @ model["W"]).argmax(axis=1)]


def _classification_report(y_true, y_pred, classes):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    print(f"  {'class':>10}  {'precision':>10}  {'recall':>7}  {'F1':>5}")
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        print(f"  {c:>10}  {p:>10.3f}  {r:>7.3f}  {f:>5.3f}")
    acc = float((y_true == y_pred).mean())
    print(f"  accuracy = {acc:.3f}")
    return acc


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    topics = {
        "sport": "score match team goal player win coach league season points".split(),
        "tech":  "server database api cloud code query script deploy bug cache".split(),
        "food":  "recipe cook flavor spice sauce bake fresh chef sweet meal".split(),
    }
    docs = []; y = []
    for label, words in topics.items():
        for _ in range(40):
            d = [words[rng.integers(len(words))] for _ in range(20)]
            docs.append(d); y.append(label)
    idx = rng.permutation(len(docs))
    docs = [docs[i] for i in idx]; y = np.array([y[i] for i in idx])
    n_train = 90
    docs_tr, docs_te = docs[:n_train], docs[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    # Multinomial NB
    m_nb = fit_multinomial_nb(docs_tr, y_tr)
    y_hat_nb = predict_nb(docs_te, m_nb)
    print("=== Multinomial Naive Bayes ===")
    _classification_report(y_te, y_hat_nb, m_nb["classes"])

    # Logistic on TF-IDF
    X_tr, vocab, idf = _tfidf(docs_tr)
    X_te, _, _ = _tfidf(docs_te + docs_tr, vocab)
    X_te = X_te[:len(docs_te)]                             # keep test rows
    # need consistent vocab & idf; simpler: refit tfidf on all docs
    X_all, vocab, idf = _tfidf(docs)
    X_tr = X_all[idx.argsort()[:n_train]]                  # unshuffled slicing tricky; use simpler:
    X_tr = X_all[:n_train]; X_te = X_all[n_train:]
    m_lr = fit_multiclass_logistic(X_tr, y_tr, l2=0.5, n_iter=400)
    y_hat_lr = predict_logistic(X_te, m_lr)
    print("\n=== Logistic regression on TF-IDF ===")
    _classification_report(y_te, y_hat_lr, m_lr["classes"])

    print("\n--- library cross-check (sklearn MultinomialNB / LogisticRegression on TF-IDF) ---")
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        cv = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None, lowercase=False)
        X_cv = cv.fit_transform([" ".join(d) for d in docs])
        clf = MultinomialNB().fit(X_cv[:n_train], y_tr)
        print(f"  sklearn NB test accuracy = {clf.score(X_cv[n_train:], y_te):.3f}")
        tf = TfidfVectorizer(tokenizer=lambda x: x.split(), token_pattern=None, lowercase=False)
        X_tf = tf.fit_transform([" ".join(d) for d in docs])
        clf2 = LogisticRegression(max_iter=1000).fit(X_tf[:n_train], y_tr)
        print(f"  sklearn LR test accuracy = {clf2.score(X_tf[n_train:], y_te):.3f}")
    except ImportError:
        print("  (sklearn not installed)")
