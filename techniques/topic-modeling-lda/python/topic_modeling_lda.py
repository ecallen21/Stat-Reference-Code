"""Latent Dirichlet Allocation via collapsed Gibbs sampling (Reference §25.4).

Generative model (Blei-Ng-Jordan 2003):
    theta_d ~ Dirichlet(alpha)           document-topic proportions
    phi_k   ~ Dirichlet(beta)             topic-word distributions
    z_{d,i} ~ Categorical(theta_d)        topic of the i-th token in doc d
    w_{d,i} ~ Categorical(phi_{z_{d,i}})  observed word

Collapsed Gibbs (Griffiths-Steyvers 2004) integrates theta and phi
analytically; at each step re-samples z for each token:

    P(z_i = k | ...) prop  (n_{d,k}^{-i} + alpha)
                          * (n_{k, w}^{-i} + beta)
                          / (n_{k}^{-i} + V * beta)

After a burn-in, average counts to estimate:
    theta_d[k] = (n_{d,k} + alpha) / (n_d + K * alpha)
    phi_k[w]   = (n_{k,w} + beta)  / (n_k + V * beta)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def fit_lda(docs, K: int, alpha: float = 0.1, beta: float = 0.01,
            n_iter: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # vocab
    vocab = {}
    for d in docs:
        for w in d:
            if w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    D = len(docs)
    # token positions
    tokens = [[vocab[w] for w in d] for d in docs]
    z = [rng.integers(0, K, size=len(t)).tolist() for t in tokens]

    n_dk = np.zeros((D, K), dtype=int)
    n_kw = np.zeros((K, V), dtype=int)
    n_k = np.zeros(K, dtype=int)
    for d in range(D):
        for i, w in enumerate(tokens[d]):
            k = z[d][i]
            n_dk[d, k] += 1
            n_kw[k, w] += 1
            n_k[k] += 1

    for it in range(n_iter):
        for d in range(D):
            for i, w in enumerate(tokens[d]):
                k = z[d][i]
                n_dk[d, k] -= 1; n_kw[k, w] -= 1; n_k[k] -= 1
                p = (n_dk[d] + alpha) * (n_kw[:, w] + beta) / (n_k + V * beta)
                p /= p.sum()
                k_new = int(rng.choice(K, p=p))
                z[d][i] = k_new
                n_dk[d, k_new] += 1; n_kw[k_new, w] += 1; n_k[k_new] += 1

    theta = (n_dk + alpha) / (n_dk.sum(axis=1, keepdims=True) + K * alpha)
    phi = (n_kw + beta) / (n_kw.sum(axis=1, keepdims=True) + V * beta)
    return {"vocab": vocab, "theta": theta, "phi": phi, "K": K,
            "n_dk": n_dk, "n_kw": n_kw,
            "method": "LDA collapsed-Gibbs (Griffiths-Steyvers 2004)"}


def top_words(model, n_top: int = 6):
    inv = {i: w for w, i in model["vocab"].items()}
    return [[inv[i] for i in np.argsort(-model["phi"][k])[:n_top]]
            for k in range(model["K"])]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # simulate 3-topic corpus with distinct vocabularies
    sport_words = "goal team match player score win coach league season".split()
    tech_words = "server database api cloud script code query bug deploy".split()
    food_words = "recipe cook flavor spice sauce bake fresh sweet chef".split()
    topic_words = [sport_words, tech_words, food_words]
    K_true = 3
    D = 80
    doc_len = 25
    docs = []
    for d in range(D):
        # each doc mixes topics with a Dirichlet prior
        theta_d = rng.dirichlet(np.array([1.0, 1.0, 1.0]) * 0.4)
        doc = []
        for _ in range(doc_len):
            k = rng.choice(K_true, p=theta_d)
            doc.append(topic_words[k][rng.integers(len(topic_words[k]))])
        docs.append(doc)

    m = fit_lda(docs, K=3, alpha=0.1, beta=0.01, n_iter=400)
    print(f"=== LDA collapsed-Gibbs: D={len(docs)}, |V|={len(m['vocab'])}, K=3 ===")
    for k, words in enumerate(top_words(m, n_top=6)):
        print(f"  topic {k}: {words}")

    print("\n--- library cross-check (sklearn.decomposition.LatentDirichletAllocation) ---")
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        vec = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None,
                              lowercase=False)
        X = vec.fit_transform([" ".join(d) for d in docs])
        lda = LatentDirichletAllocation(n_components=3, learning_method="batch",
                                          random_state=0, max_iter=50)
        lda.fit(X)
        vocab_sk = vec.get_feature_names_out()
        for k, comp in enumerate(lda.components_):
            top = [vocab_sk[i] for i in np.argsort(-comp)[:6]]
            print(f"  sklearn topic {k}: {top}")
    except ImportError:
        print("  (sklearn not installed)")
