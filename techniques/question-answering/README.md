# Question Answering (Reference §25.x extra)

Given a passage `P` and a question `Q`, produce an answer either by
**extracting** a span from `P` (span selection) or **generating** free text
(abstractive / RAG).

## Extractive QA — span selection

Predict `(start, end)` positions in the passage. Modelled as two token
classification heads on top of a contextual encoder:

```
logits_start[t] = w_s · h_t
logits_end[t]   = w_e · h_t
loss = CE(y_start, softmax(logits_start)) + CE(y_end, softmax(logits_end))
```

Sweep over `(i, j)` with `i ≤ j ≤ i + max_len` and pick the pair with the
maximum joint score. This is the workhorse SQuAD-style model.

## Abstractive / RAG

- **Abstractive**: seq2seq generator (T5, BART) conditioned on `Q + P` produces the answer as free text; can rephrase and combine.
- **RAG (retrieval-augmented generation)**: retrieve top-k passages from a corpus, feed to an LLM as context, generate the answer. Standard for open-domain QA.
- **LLM + tool use**: ReAct, Toolformer — the LM decides when to call a search API, calculator, or knowledge base.

## When to use

- **Extractive** — when the answer is a literal span in the passage; used in factoid QA, reading comprehension, contract analysis.
- **RAG** — open-domain QA over a document store (support wiki, medical records, legal corpus, product manuals).
- **Multi-hop** — questions requiring reasoning across multiple passages (HotpotQA, StrategyQA); chain-of-thought / tool-use LMs.
- **NOT** for opinion / hallucination-risky questions without a grounding source.

## Files

- `python/question_answering.py` — from-scratch IDF-weighted **sentence-retrieval baseline** (the retrieve step of a real system). Demo on a Marie Curie passage with 4 questions: 2/4 correct on this hard toy. The failures ("husband Pierre Curie", "radioactivity") happen because the answer sentence doesn't repeat "Marie Curie", so lexical overlap misses them — exactly why production QA uses neural encoders that understand coreference and semantics.
- `r/question_answering.R` — `reticulate` + `huggingface transformers.pipeline('question-answering')`, `haystack`, `langchain`, `llama-index`.

## Assumptions & caveats

- **Extractive assumes** the answer is exactly in the passage — abstract paraphrases fail.
- **Multiple answers** — modern QA models can return the top-k spans with probabilities; use for disambiguation.
- **Unanswerable questions** — SQuAD 2.0 introduced "no answer" as a legitimate label; predict a null span with high probability.
- **Long documents** — passage may exceed the context; use sliding windows or hierarchical retrieval + reader.
- **Numeric / discrete answers** — often better handled by a generator + calculator tool than by span selection.
- **RAG hallucination** — retriever misses the passage → LM invents an answer. Cite sources and detect low-support answers.

## Related in this repo

- `tfidf-bm25`, `sentence-similarity` — retriever building blocks.
- `named-entity-recognition`, `coreference-resolution`, `word-sense-disambiguation` — upstream comprehension tasks.
- `transformer-encoder`, `masked-language-modeling` — the encoder families used for QA.
- `textrank-summarization`, `abstractive-summarization` — sibling long-form tasks.

## Run

```
python techniques/question-answering/python/question_answering.py
Rscript techniques/question-answering/r/question_answering.R
```

**Refs:** Rajpurkar, P. et al. "SQuAD: 100,000+ questions for machine comprehension of text." *EMNLP*, 2016; Devlin, J. et al. "BERT: pre-training of deep bidirectional transformers for language understanding." *NAACL*, 2019; Lewis, P. et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks (RAG)." *NeurIPS*, 2020.

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
