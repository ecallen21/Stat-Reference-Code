# Context Compression / RECOMP (Reference §47.193)

Xu, Shi & Choi (2024, ICLR). Retrieved passages are often long and
noisy; RECOMP inserts a **compressor** between retrieval and
generation:

    docs (top-k, long) → Compressor → summary_i or FILTERED subset
    → LLM(query, summary/subset)

Two flavours:
- **Extractive**: score each sentence, keep the top-p.
- **Abstractive**: fine-tuned encoder-decoder summariser.

Compresses long context 2-10× with minimal / no answer-quality
drop, cutting inference cost dramatically.

## Files

- `python/context_compression_recomp.py` — extractive-compression
  demo on a 3-passage corpus with 3 queries. Cuts context from
  80 → 15-29 tokens (~3-4× smaller); preserves the correct
  answerable sentence for all three queries.
- `r/context_compression_recomp.R` — no R port; recommends
  `carriex/recomp`, `llmlingua`, `langchain.LLMChainExtractor`.

## When to use

- **Long-document RAG** — feeding 20 × 500-word passages to an
  LLM is expensive.
- **Latency-critical RAG** — compression cuts prompt-processing
  time.
- **Compact-context LLMs** (long prompts blow up cost) —
  compression = cost saving.

## When NOT to use

- **When passages are already short and relevant** — compression
  adds noise.
- **Tasks needing exact quotes** — extractive compression drops
  most sentences, may miss the exact match.
- **Multi-hop reasoning** — the compressor may drop the "bridge"
  passage.

## Assumptions & caveats

- **Compressor quality** — a bad compressor drops the answer;
  measure end-to-end.
- **Selective augmentation** — RECOMP's contribution: also decide
  WHEN to retrieve at all.
- **Abstractive vs extractive** — abstractive is more fluent but
  can hallucinate; extractive is safer.
- **Compression ratio** — 2-10× typical; more aggressive
  compressors need domain fine-tuning.

## Related in this repo

- `retrieval-augmented-generation`, `self-rag`,
  `hyde-hypothetical-doc` — RAG pipeline neighbours.
- `dense-passage-retrieval-dpr`, `colbert-late-interaction`,
  `cross-encoder-reranker` — retrieval-stage neighbours.
- `abstractive-summarization`, `question-answering` — related
  NLP tasks the compressor is analogous to.

## Run

```
python techniques/context-compression-recomp/python/context_compression_recomp.py
Rscript techniques/context-compression-recomp/r/context_compression_recomp.R
```

**Refs:** Xu, F., Shi, W. & Choi, E. "RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation." *ICLR*, 2024; Jiang, H. et al. "LLMLingua: Compressing prompts for accelerated inference of large language models." *EMNLP*, 2023.

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
