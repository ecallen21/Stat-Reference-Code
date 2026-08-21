# CKY Chart Parser (Reference §25.x extra)

Dynamic-program parser for a **context-free grammar in Chomsky Normal Form (CNF)**:

```
A → B C      binary rule
A → w        terminal / lexical rule
```

## Algorithm

`table[i][j][A]` = max probability of an `A`-headed tree spanning tokens `[i, j)`. Fill by increasing span length:

```
lex:    table[i][i+1][A]  = P(A → w_i)  for every A → w_i rule
binary: table[i][k][A]    = max_{B, C, mid} P(A → B C) · table[i][mid][B] · table[mid][k][C]
```

Back-pointers recover the highest-scoring parse (**Viterbi**). Time and space are `O(n³ · |grammar|)`.

## When to use

- **Probabilistic constituency parsing** with a hand-authored PCFG (old-school NLP, teaching).
- **Formal-language recognition** — SQL, mathematical expressions, DSLs.
- **Speech understanding** — pre-neural systems used a PCFG over ASR lattices.
- **Modern practice**: neural constituency (**benepar** — Kitaev-Klein 2018) or dependency parsers (spaCy, stanza, trankit) fine-tune from BERT / T5; PCFG parsers are pedagogical or legacy.

## Files

- `python/syntactic_parsing_cky.py` — from-scratch Viterbi CKY on a PCFG in CNF. Toy grammar with S/NP/VP/PP/DET/N/V/P; parses the classic PP-attachment ambiguity sentence "the cat saw the dog with the telescope" and returns the VP-attachment reading (`(VP (VP (V saw) (NP the dog)) (PP (P with) (NP the telescope)))`) as the highest-probability parse under the given rule weights. Log-prob −6.83.
- `r/syntactic_parsing_cky.R` — `NLP + openNLP::Parse_Annotator`, `udpipe::udpipe_annotate(parser='default')`, `spacyr::spacy_parse(dependency=TRUE)`; Python `nltk.ChartParser / ViterbiParser`, `benepar`, `stanza`, `spacy`.

## Assumptions & caveats

- **CNF conversion** — any CFG can be converted to CNF (Chomsky 1959); non-CNF grammars can be parsed by **Earley** (`O(n³)` for arbitrary CFGs, faster for many).
- **PCFG independence** — probabilities of subtrees are independent given the head; misses long-range regularities. **Lexicalised** PCFGs (Collins, Charniak) add head-word features and were pre-neural SOTA.
- **Ambiguity** is often extreme — dozens of parses per sentence; probabilities disambiguate but training data is expensive.
- **Grammar induction** from raw text is possible (grammar-EM / inside-outside) but rarely useful past small toy examples.
- **Modern SOTA** — Kitaev & Klein 2018 (benepar) hit ~96 F1 on Penn Treebank; dependency parsing has similar accuracy and is preferred for most downstream NLP.

## Related in this repo

- `pos-tagging` — the tagger providing part-of-speech input to a parser.
- `hmm`, `named-entity-recognition` — sequence-level cousins.
- `ngram-language-model` — an alternative sequence model with no tree structure.

## Run

```
python techniques/syntactic-parsing-cky/python/syntactic_parsing_cky.py
Rscript techniques/syntactic-parsing-cky/r/syntactic_parsing_cky.R
```

**Refs:** Kasami, T. "An efficient recognition and syntax algorithm for context-free languages." AFCRL-65-758, 1965; Younger, D.H. "Recognition and parsing of context-free languages in time n³." *Information and Control* 10, 189–208, 1967; Charniak, E. "A maximum-entropy-inspired parser." *NAACL*, 2000; Kitaev, N. & Klein, D. "Constituency parsing with a self-attentive encoder." *ACL*, 2018.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
