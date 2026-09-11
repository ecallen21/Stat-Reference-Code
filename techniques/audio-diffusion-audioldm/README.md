# Audio Diffusion / AudioLDM (Reference §47.230)

Liu et al. (2023, ICML). Diffusion in a **mel-spectrogram-based
latent space**, decoded by a HiFi-GAN vocoder:

1. Waveform → mel-spectrogram (2-D 'image').
2. VAE encodes mel-spec to latent z.
3. Diffusion model in z-space, text-conditioned via CLAP encoder.
4. Decode z → mel-spec → waveform (vocoder).

Enables text-to-audio: 'sound of thunder', 'wooden creaking'.
Similar tech: StableAudio, MusicGen.

## Files

- `python/audio_diffusion_audioldm.py` — toy mel-spec + linear
  denoise + Griffin-Lim-style vocoder on a 1-s / 8 kHz waveform:
  - Noisy spec MSE 3.95 → **denoised 0.06** (98.5 % reduction).
  - Reconstruction SNR shown as illustrative (real vocoders do
    much better).
- `r/audio_diffusion_audioldm.R` — recommends
  `haoheliu/AudioLDM`, `audiocraft`, `stable-audio-tools`.

## When to use

- **Text-to-audio / music generation**.
- **Sound-effects libraries** for games / film.
- **Speech synthesis** — related pipelines (Tortoise-TTS, Bark,
  XTTS).

## When NOT to use

- **When exact spectral fidelity** matters (music-production
  mastering) — diffusion is lossy.
- **Real-time interactive** — even 10-s clips take seconds on GPU.
- **Very short SFX** (< 1 s) — cascaded models overkill.

## Assumptions & caveats

- **Vocoder quality** determines waveform naturalness (HiFi-GAN
  vs BigVGAN vs Encodec).
- **CLAP encoder** for text conditioning — analogous to CLIP for
  audio.
- **Sample rate** typically 16-48 kHz; higher rates need bigger
  models.
- **Latent duration** limits generation length.

## Related in this repo

- `diffusion-model`, `latent-diffusion-ldm`,
  `ddim-implicit-diffusion`, `classifier-free-guidance` —
  diffusion cousins.
- `speech-recognition-whisper`, `wav2vec-ssl-audio` — related
  audio-model neighbours.

## Run

```
python techniques/audio-diffusion-audioldm/python/audio_diffusion_audioldm.py
Rscript techniques/audio-diffusion-audioldm/r/audio_diffusion_audioldm.R
```

**Refs:** Liu, H. et al. "AudioLDM: Text-to-audio generation with latent diffusion models." *ICML*, 2023; Copet, J. et al. "Simple and controllable music generation (MusicGen)." *NeurIPS*, 2023.

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
