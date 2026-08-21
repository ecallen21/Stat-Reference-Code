# Named-Entity Recognition (Reference §25.8)

Sequence-tagging task: assign each token in a sentence a label from a small
inventory (e.g. **B-PER** / **I-PER**, **B-ORG** / **I-ORG**, **O** for
outside). Standard entity types: PER (person), ORG (organisation), LOC
(location), GPE (geo-political entity), DATE, MONEY.

## BIO / IOB2 tag scheme

- `B-X` — beginning of an entity of type X.
- `I-X` — inside (continuation) of an entity of type X.
- `O` — outside any entity.

Variants: **BIOES** adds `E-` (end) and `S-` (singleton) for richer segmentation.

## Models

| Family | Where used |
|---|---|
| **Dictionary / gazetteer lookup** | Cold-start baseline; recall depends on the dictionary. |
| **HMM with Viterbi decoding** | Classical statistical baseline (this module). |
| **Linear-chain CRF** | Better feature integration; standard pre-2016. `crfsuite`, `sklearn-crfsuite`. |
| **BiLSTM-CRF** | Lample et al. 2016; strong until transformers took over. |
| **BERT + linear head (or CRF)** | Current default; fine-tune a pretrained model. |
| **Transformer-based zero-shot NER** | `spacy` v3+ or `flair` with `xlm-roberta-large`; no training needed. |

## When to use

- **Downstream NLP** — populate knowledge graphs, resolve mentions, prep for coreference / relation extraction.
- **Redaction / de-identification** — flag PER / ORG / LOC to remove.
- **Search enrichment** — index documents by mentioned entities.
- **Analytics** — count mentions of entities over time.

## Files

- `python/named_entity_recognition.py` — from-scratch HMM NER with Viterbi decoding, trained on tiny labelled corpus. Demo recognises PER (Alice / Bob / David) and ORG (Acme Corp / Globex Industries / Initech) in unseen sentences; entity-level F1 = 0.875 on the training set. Includes a small entity-span F1 utility.
- `r/named_entity_recognition.R` — `crfsuite::crf`, `spacyr::spacy_parse(entity=TRUE)`, `udpipe::udpipe_annotate`; Python `spacy`, `flair`, `transformers.pipeline('ner')`.

## Assumptions & caveats

- **Small labelled data is fragile** — HMMs / CRFs need thousands of tagged sentences to generalise. Transformers pretrained on general text bring most of the knowledge.
- **Out-of-vocabulary words** — the HMM here uses a uniform emission for unseen words; classical mitigations: word-shape features (capitalisation, digits), character n-grams, POS tags.
- **Nested / overlapping entities** — BIO doesn't handle "New York" being both LOC and part of "New York Times"; use span-based models or joint parsing.
- **Domain shift** — a Wikipedia-trained NER underperforms on clinical / legal text; fine-tune on domain corpus.
- **Entity linking is a separate step** — mapping "Apple" (mentioned) → Q312 (Wikidata) or a knowledge base is downstream of NER.
- **Evaluation is span-based** — token-level accuracy overstates quality because O dominates.

## Related in this repo

- `hmm` — the general HMM machinery used here.
- `text-classification` — coarse-document-level analogue.
- `text-preprocessing` — tokenisation.

## Run

```
python techniques/named-entity-recognition/python/named_entity_recognition.py
Rscript techniques/named-entity-recognition/r/named_entity_recognition.R
```

**Refs:** Rabiner, L.R. "A tutorial on hidden Markov models and selected applications in speech recognition." *Proc. IEEE* 77(2), 257–286, 1989; Lafferty, J., McCallum, A. & Pereira, F. "Conditional random fields: probabilistic models for segmenting and labeling sequence data." *ICML*, 2001; Lample, G. et al. "Neural architectures for named entity recognition." *NAACL*, 2016.

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
