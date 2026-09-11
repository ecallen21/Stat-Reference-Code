# wav2vec 2.0 — Self-Supervised Speech Representations (Reference §47.232)

Baevski, Zhou, Mohamed & Auli (2020, NeurIPS). Two-stage:

1. **CNN feature encoder**: raw waveform → latent z_1..z_T.
2. **Transformer context net**: mask spans of z_t and predict via
   CONTRASTIVE loss against QUANTISED targets q_t (small learned
   codebook via Gumbel-softmax).

    L_contrastive = −log( exp(sim(c_t, q_t) / κ) /
                          Σ_qhat exp(sim(c_t, qhat) / κ) )

Enables strong ASR with **10 min of labelled data + 60k hours of
unlabelled** (LibriLight). Direct ancestor of Whisper's audio
encoder philosophy.

## Files

- `python/wav2vec_ssl_audio.py` — toy CNN encoder + codebook +
  masked contrastive loss on a 1-s / 8 kHz waveform (80 frames):
  - Masked 32/80 frames; **loss with correct targets 2.58**
    vs **random targets 4.59** (contrast works).
  - Illustrates encoder + quantiser + contrastive pipeline.
- `r/wav2vec_ssl_audio.R` — recommends
  `facebookresearch/fairseq` wav2vec, `transformers.Wav2Vec2Model`.

## When to use

- **Low-resource ASR** — fine-tune with tiny labelled sets.
- **Speech embedding** extraction for downstream classification.
- **Audio-domain pretraining** for other tasks (emotion, speaker
  ID).

## When NOT to use

- **When Whisper is enough** — Whisper is often simpler for ASR.
- **Very long-form audio** — wav2vec 2.0 has quadratic attention.
- **Music / non-speech domains** — designed for speech.

## Assumptions & caveats

- **Codebook size** and Gumbel temperature control quantiser
  quality.
- **Masking spans** (span_len 3-10) essential — too-short spans
  are trivial.
- **CTC head** for ASR fine-tuning; letter/phoneme output.
- **HuBERT** (Hsu 2021) is a related MSM-style variant.

## Related in this repo

- `speech-recognition-whisper` — downstream ASR cousin.
- `contrastive-learning`, `simclr-contrastive`,
  `barlow-twins`, `dino-self-supervised-vision` — SSL cousins
  in other modalities.
- `audio-diffusion-audioldm` — audio-generation neighbour.

## Run

```
python techniques/wav2vec-ssl-audio/python/wav2vec_ssl_audio.py
Rscript techniques/wav2vec-ssl-audio/r/wav2vec_ssl_audio.R
```

**Refs:** Baevski, A., Zhou, H., Mohamed, A. & Auli, M. "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS*, 2020; Hsu, W.-N. et al. "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units." *IEEE/ACM TASLP*, 2021.

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
