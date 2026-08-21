"""CKY chart parser for probabilistic CFG in Chomsky Normal Form (Reference §25.x extra).

Grammar in CNF:
    A -> B C          (binary rule)
    A -> w             (terminal / lexical rule)

CKY dynamic program on the O(n^3) chart:
    table[i][j][A] = max prob of an A-spanning-tree for tokens [i, j)
    table[i][k][A] = max_{B, C, mid} P(A -> B C) * table[i][mid][B] * table[mid][k][C]

Viterbi backpointers recover the highest-scoring parse tree.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import defaultdict    # stdlib

import numpy as np    # numerical arrays + linear algebra


def cky_parse(tokens, binary_rules: dict, lex_rules: dict,
              start: str = "S") -> dict:
    """binary_rules[(A, B, C)] = prob;  lex_rules[(A, w)] = prob."""
    n = len(tokens)
    # table[i][j] = {non-terminal: (log_prob, backpointer)}
    table = [[defaultdict(lambda: (-math.inf, None)) for _ in range(n + 1)] for _ in range(n + 1)]
    # lexical layer
    for i, w in enumerate(tokens):
        for (A, ww), p in lex_rules.items():
            if ww == w:
                table[i][i + 1][A] = (math.log(p + 1e-12), ("term", w))
    # binary rules by span length
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            k = i + span
            for mid in range(i + 1, k):
                for (A, B, C), p in binary_rules.items():
                    lb, _ = table[i][mid][B]; lc, _ = table[mid][k][C]
                    if lb == -math.inf or lc == -math.inf:
                        continue
                    score = math.log(p + 1e-12) + lb + lc
                    if score > table[i][k][A][0]:
                        table[i][k][A] = (score, ("bin", B, C, mid))
    # recover tree from table[0][n][start]
    def _tree(i, j, A):
        best, bp = table[i][j][A]
        if bp is None:
            return None
        if bp[0] == "term":
            return (A, bp[1])
        _, B, C, mid = bp
        return (A, _tree(i, mid, B), _tree(mid, j, C))
    tree = _tree(0, n, start)
    return {"tree": tree, "log_prob": table[0][n][start][0]}


def _pprint(tree, indent: int = 0) -> str:
    if tree is None:
        return "  " * indent + "(no parse)"
    if len(tree) == 2:
        return "  " * indent + f"({tree[0]} {tree[1]})"
    lines = ["  " * indent + f"({tree[0]}"]
    lines.append(_pprint(tree[1], indent + 1))
    lines.append(_pprint(tree[2], indent + 1))
    return "\n".join(lines) + ")"


if __name__ == "__main__":
    # Toy CNF grammar: NP + VP -> S; NP + PP -> NP; VP + PP -> VP; V + NP -> VP; P + NP -> PP.
    binary = {
        ("S", "NP", "VP"): 1.0,
        ("VP", "V", "NP"): 0.6,
        ("VP", "VP", "PP"): 0.4,
        ("NP", "DET", "N"): 0.5,
        ("NP", "NP", "PP"): 0.2,
        ("NP", "N", "N"): 0.3,
        ("PP", "P", "NP"): 1.0,
    }
    lex = {
        ("DET", "the"): 1.0,
        ("N", "cat"): 0.4,
        ("N", "dog"): 0.3,
        ("N", "telescope"): 0.3,
        ("V", "saw"): 1.0,
        ("P", "with"): 1.0,
    }
    tokens = "the cat saw the dog with the telescope".split()
    res = cky_parse(tokens, binary, lex, start="S")
    print("=== CKY parse (probabilistic CFG in CNF) ===\n")
    print(f"  tokens: {tokens}")
    print(f"  log-prob of MAP parse: {res['log_prob']:.4f}\n")
    print(_pprint(res["tree"]))

    print("\n--- library cross-check (nltk.CFG + nltk.CkyParser; berkeley-parser; benepar) ---")
