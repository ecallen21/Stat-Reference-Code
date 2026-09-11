# Whisper — Speech Recognition (Reference §47.231)

Radford, Kim, Xu, Brockman, McLeavey & Sutskever (2022, OpenAI /
Interspeech 2023). Encoder-decoder Transformer trained on
**680,000 hours** of weakly-labelled internet audio:

    Audio → log-Mel spectrogram (80 × T)
    → Transformer ENCODER (self-attn over time)
    → Transformer DECODER (cross-attn to encoder + causal
      self-attn over tokens)
    → transcription in the language's script with punctuation.

Handles 99 languages, translates to English, timestamps and
paralinguistic tokens ([BGM], [applause]).

## Files

- `python/speech_recognition_whisper.py` — toy mel-spec +
  encoder + decoder with LANG / TASK special tokens. Real Whisper
  uses a proper LM head trained on huge audio corpora; this demo
  only shows the pipeline structure.
- `r/speech_recognition_whisper.R` — recommends
  `openai-whisper`, `faster-whisper`, `distil-whisper`,
  `transformers.WhisperForConditionalGeneration`.

## When to use

- **Multilingual ASR** — Whisper large-v3 covers 99 languages.
- **Zero-shot translation** to English.
- **Long-form transcription** with automatic timestamps.

## When NOT to use

- **Real-time / streaming** (baseline Whisper is offline; use
  streaming variants).
- **Very low-resource languages** not in training.
- **Domain-specific vocabulary** (medical / legal) without
  fine-tuning.

## Assumptions & caveats

- **Model sizes**: tiny (39 M) → large-v3 (1.5 B); larger =
  slower but more accurate.
- **Prompt tokens** — LANG / TASK / TIMESTAMPS steer output
  format.
- **Chunking** — 30-second windows; use overlap for continuity.
- **Hallucinations** in silent / noisy segments — filter with
  VAD.

## Related in this repo

- `wav2vec-ssl-audio` — related SSL audio encoder ancestor.
- `audio-diffusion-audioldm` — audio-generation cousin.
- `transformer-encoder`, `transformer-decoder`,
  `attention-mechanism` — architecture neighbours.

## Run

```
python techniques/speech-recognition-whisper/python/speech_recognition_whisper.py
Rscript techniques/speech-recognition-whisper/r/speech_recognition_whisper.R
```

**Refs:** Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. & Sutskever, I. "Robust speech recognition via large-scale weak supervision (Whisper)." *arXiv:2212.04356*, 2022.

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
