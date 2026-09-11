"""Byte-Pair Encoding - BPE Tokenization (Reference Sec 47.159).

Gage 1994 'A New Algorithm for Data Compression', C Users J;
Sennrich, Haddow & Birch 2016 'Neural Machine Translation of Rare
Words with Subword Units', ACL. Learns a merge table:

    1. Start with byte / character-level vocabulary.
    2. Count adjacent-symbol pairs; merge the most frequent pair.
    3. Repeat until target vocab size or merge budget is reached.

Encoding: repeatedly apply learned merges to the input string.
Yields sub-word units that generalise to unseen / rare words.
Backbone of GPT / RoBERTa / T5 tokenizers (byte-level BPE).
"""
from __future__ import annotations    # stdlib

from collections import Counter    # frequency counting

import numpy as np    # numerical arrays


def learn_bpe(corpus, n_merges=100):
    """Learn BPE merges from a token frequency dict.

    Corpus is a dict {word_str: count} where words are pre-split as space-
    separated character sequences with a trailing '</w>' end-of-word marker.
    """
    vocab = dict(corpus)
    merges = []
    for step in range(n_merges):
        pairs = Counter()
        for word, cnt in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += cnt
        if not pairs: break
        best = max(pairs.items(), key=lambda kv: kv[1])[0]
        merges.append(best)
        # Apply merge across all words
        new_vocab = {}
        bigram = " ".join(best)
        replacement = "".join(best)
        for word, cnt in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = cnt
        vocab = new_vocab
    return merges, vocab


def apply_bpe(word, merges):
    """Encode a single word (character-tokenised with </w>) via learned merges."""
    symbols = list(word) + ["</w>"]
    text = " ".join(symbols)
    for a, b in merges:
        text = text.replace(f"{a} {b}", f"{a}{b}")
    return text.split()


if __name__ == "__main__":
    print("=== Byte-Pair Encoding (Gage 1994; Sennrich et al 2016) ===\n")

    # Toy corpus: nature words with common prefixes / suffixes
    corpus_words = ["lower", "lowest", "newer", "newest", "widest", "wide",
                     "wider", "widening", "widened", "narrow", "narrower",
                     "narrowest", "narrowing"] * 3
    # Convert to BPE-friendly space-separated char-form with </w>
    counter = Counter(corpus_words)
    vocab_init = {" ".join(list(w)) + " </w>": c for w, c in counter.items()}

    print("  Corpus size:", sum(counter.values()), "tokens,",
            len(counter), "unique words")

    for n_merges in [5, 20, 50]:
        merges, _ = learn_bpe(vocab_init, n_merges=n_merges)
        # Encode two unseen words
        enc_widen = apply_bpe("widen", merges)
        enc_newest = apply_bpe("newest", merges)
        enc_rare = apply_bpe("narrowly", merges)
        print(f"\n  n_merges = {n_merges}")
        print(f"    'widen'    -> {enc_widen}   (# tokens = {len(enc_widen)})")
        print(f"    'newest'   -> {enc_newest}   (# tokens = {len(enc_newest)})")
        print(f"    'narrowly' -> {enc_rare}   (# tokens = {len(enc_rare)})")

    # Compression check: total tokens after 50 merges vs char count
    merges, vocab_final = learn_bpe(vocab_init, n_merges=50)
    total_chars = sum(len(w) * counter[w] for w in counter)
    total_bpe = sum(len(apply_bpe(w, merges)) * counter[w] for w in counter)
    print(f"\n  Total char tokens (baseline): {total_chars}")
    print(f"  Total BPE tokens (50 merges): {total_bpe}   "
          f"({100 * (1 - total_bpe / total_chars):.1f}% reduction)")

    print("\n--- library cross-check (sentencepiece / huggingface tokenizers Python) ---")
