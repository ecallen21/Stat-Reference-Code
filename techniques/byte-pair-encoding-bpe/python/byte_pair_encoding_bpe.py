"""Byte-Pair Encoding tokenisation (Reference Sec 47.83).

Sennrich, Haddow & Birch 2016 'Neural machine translation of rare
words with subword units', ACL. Adapted from Gage 1994 for
compression. Start with a character-level vocabulary; repeatedly

    1. count all adjacent symbol pairs in the corpus,
    2. merge the most frequent pair into a new symbol,

until vocab size hits the target. Handles morphology + unknowns
gracefully (rare words fall back to characters); the workhorse
tokeniser of GPT / BERT-style LLMs (Radford GPT-2 uses byte-level
BPE over UTF-8 bytes).
"""
from __future__ import annotations    # stdlib

from collections import Counter    # pair frequency
from typing import List, Tuple    # type hints


def _get_pair_freqs(splits, weights):
    freq = Counter()
    for tokens, w in zip(splits, weights):
        for a, b in zip(tokens, tokens[1:]):
            freq[(a, b)] += w
    return freq


def _merge(splits, pair):
    merged = []
    for tokens in splits:
        i = 0; out = []
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                out.append(tokens[i]); i += 1
        merged.append(out)
    return merged


def bpe_train(corpus, n_merges):
    """corpus is dict[word -> count]. Returns list of merges."""
    splits = [list(w) + ["</w>"] for w in corpus.keys()]
    weights = list(corpus.values())
    merges = []
    for step in range(n_merges):
        f = _get_pair_freqs(splits, weights)
        if not f: break
        top = max(f, key=f.get)
        merges.append(top)
        splits = _merge(splits, top)
    return merges


def bpe_apply(word, merges):
    tokens = list(word) + ["</w>"]
    for pair in merges:
        i = 0; out = []
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                out.append(tokens[i]); i += 1
        tokens = out
    return tokens


if __name__ == "__main__":
    print("=== Byte-Pair Encoding (Sennrich et al 2016) ===\n")
    corpus = {"low": 5, "lowest": 2, "newer": 6, "wider": 3, "new": 4}
    for n in [4, 10, 20]:
        merges = bpe_train(corpus, n_merges=n)
        print(f"  After {len(merges)} merges (target {n}):")
        for w in corpus:
            print(f"    {w:8s} -> {bpe_apply(w, merges)}")

        # Unknown word tokenisation
        print(f"    UNK 'lowering' -> {bpe_apply('lowering', merges)}")
        print()

    print("  BPE learns morpheme-like units ('low', 'er', 'est', 'new') from")
    print("  co-occurrence -- no linguistic knowledge required.")
    print("\n--- library cross-check (tokenizers HF Python; text2vec R) ---")
