# Abstractive Summarisation (Reference §25.x extra)

Generate a summary in **new words** — may reword, combine, or omit — as
opposed to `textrank-summarization` which extracts sentences verbatim.

## Standard architecture

**Encoder-decoder transformer** trained on `(article, summary)` pairs:

- **T5** (Raffel 2020), **BART** (Lewis 2020), **Pegasus** (Zhang 2020), **mT5**, **mBART**, **DistilBART**.
- Trained by teacher-forced cross-entropy on gold summaries.
- Decoded with beam search (`num_beams=4`) + length penalty + no-repeat-n-gram blocking.
- Modern default: fine-tune BART / T5 or prompt an LLM (`gpt-4`, `claude`, `gemini`).

## Extractive vs abstractive vs hybrid

| Approach | Faithfulness | Fluency |
|---|---|---|
| **Extractive** (`textrank-summarization`) | high | choppy |
| **Abstractive** (seq2seq transformer) | can hallucinate | fluent |
| **Hybrid** (extract-then-simplify, this demo) | between | between |
| **LLM prompting** | depends on prompt / model | very fluent, hallucination risk |

## When to use

- **News summarisation** — CNN/DailyMail benchmarks.
- **Meeting / dialogue summarisation** — MediaSum, DialogSum.
- **Scientific abstracts** — arXiv-abstract, PubMed.
- **Product / support summarisation** — internal knowledge bases.
- **Legal / medical** — HIGH-STAKES, need faithfulness-checked pipelines; hallucination is a legal / clinical risk.

## Files

- `python/abstractive_summarization.py` — from-scratch extract-then-simplify baseline: run TextRank (PageRank on the TF-IDF sentence-similarity graph) to pick top-k sentences, then apply rule-based simplifications (drop parentheticals, dashed asides). Demo on a 5-sentence renewable-energy article: picks the top 3 sentences and strips "(from the European Union to India and China)", "-- reaching about 30% of the global electricity mix in 2023", and "(both intermittent)". Cleaner than pure extractive, without hallucination risk.
- `r/abstractive_summarization.R` — `reticulate` + Python `huggingface transformers.pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')`; canonical models BART, Pegasus, T5, FLAN-T5.

## Assumptions & caveats

- **Hallucination** is the perennial risk of true abstractive models. Faithfulness metrics (SummaC, QAGS, FactCC) exist but are imperfect.
- **Length control** — the model may under- or over-shoot; length penalty + `max_new_tokens` help.
- **Copy mechanisms** (See 2017) — force the decoder to copy rare / OOV tokens from the source; useful for named entities.
- **RAG-style summarisation** — retrieve relevant sentences then let the LM summarise; grounds the output.
- **Multi-document summarisation** — concatenate + attention masks; requires cross-document coreference.
- **Query-focused summarisation** — summary conditioned on a user question; combines with QA.

## Related in this repo

- `textrank-summarization`, `bleu-rouge-eval`, `bertscore-chrf-metrics` — sibling techniques + evaluation.
- `transformer-encoder`, `transformer-decoder`, `text-generation-decoding` — the seq2seq machinery.
- `question-answering` — the query-focused summarisation neighbour.

## Run

```
python techniques/abstractive-summarization/python/abstractive_summarization.py
Rscript techniques/abstractive-summarization/r/abstractive_summarization.R
```

**Refs:** See, A. et al. "Get to the point: summarization with pointer-generator networks." *ACL*, 2017; Lewis, M. et al. "BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension." *ACL*, 2020; Zhang, J. et al. "PEGASUS: pre-training with extracted gap-sentences for abstractive summarization." *ICML*, 2020.

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
