# Quantisation and Pruning (Reference §27.x extra)

Two workhorse model-compression techniques. Deploy the same trained model
at a fraction of the memory / latency, usually with a small quality tax.

## Quantisation

Represent weights (and often activations) with lower-precision integers.

- **Symmetric int8**: `x_q = round(x / scale)`, `scale = max(|x|) / 127`. Zero maps to zero.
- **Asymmetric uint8**: `x_q = round((x − zp) / scale)`. Handles skewed distributions better.
- **Per-tensor / per-channel** — one scale for the whole tensor, or one per output channel; per-channel usually better.
- **Post-training** (this module) vs **quantisation-aware training (QAT)**.
- **Modern LLM quantisation**: bitsandbytes int8 / int4, GPTQ, AWQ, QLoRA (4-bit + LoRA fine-tuning).

## Pruning

Zero out some fraction of weights.

- **Unstructured / magnitude**: threshold on `|w|`; irregular sparsity. Only helps memory unless sparse kernels are used.
- **Structured**: whole rows / columns / channels / attention heads. Straightforward speed-up on any hardware.
- **2:4 structured sparsity** (NVIDIA Ampere+): every 4 consecutive weights have exactly 2 non-zero. 1.5× throughput at the same memory.
- **Iterative magnitude pruning** (Han 2015) — retrain after each pruning round; recovers much of the lost accuracy.
- **Lottery ticket hypothesis** (Frankle-Carbin 2019) — a sparse initialisation exists that trains from scratch to full accuracy.

## When to use

- **Edge / mobile deployment** — int8 on Apple ANE / Hexagon / Coral / Jetson.
- **LLM inference** — int8 / int4 quantisation drops memory 4–8× at ~1% task-quality cost.
- **CPU inference** — int8 gives 2–4× throughput vs fp32 on x86 / ARM.
- **Sparse-kernel-supporting GPUs** (Ampere+): 2:4 sparsity + int8 for peak throughput.
- **NOT training** — training below fp16 is possible (fp8, bf16) but needs care.

## Files

- `python/quantization_pruning.py` — from-scratch symmetric int8, asymmetric uint8, magnitude pruning, and structured row pruning. Demo on a 64×128 Gaussian weight matrix:
  - **int8 sym**: RMSE 0.0089, memory 65 KB → 8 KB (8×).
  - **uint8 asym**: RMSE 0.0081 (marginally better because asymmetric uses full 256 range).
  - **Pruning**: sparsity 0.5 → 50% zeros, RMSE 0.27.
  - **Structured** row-pruning at 50%: RMSE 0.68 (more aggressive; keeps compute regular).
  - **Combined 50% prune + int8**: RMSE 0.27, memory ~1/8.
- `r/quantization_pruning.R` — `torch::.quantize_per_tensor`, `torch::nn_utils_prune_*`; Python `torch.quantization`, `bitsandbytes`, GPTQ, AWQ, QLoRA.

## Assumptions & caveats

- **Post-training quantisation is lossy** — some layers (norm layers, embeddings) shouldn't be quantised; QAT recovers most of the drop.
- **Activation quantisation** is harder than weight quantisation because activations are input-dependent; calibrate with a representative dataset.
- **Outlier handling** matters at low bits — a few outlier activations wreck int8 quality (Dettmers 2022); llm-int8 keeps outliers in fp16.
- **Unstructured sparsity requires sparse kernels** — without them you get memory savings only, not speed-ups.
- **Pruning + quantisation compose** — do both, but re-tune the quantisation scales after pruning.
- **Fine-tuning after pruning** (Han 2016; lottery ticket) recovers most of the accuracy drop.

## Related in this repo

- `knowledge-distillation` — orthogonal compression technique; combine for maximum shrinkage.
- `mixture-of-experts` — a compute-parallel compression via sparse routing.
- `adam-optimizer` — training-loop counterpart when QAT is used.

## Run

```
python techniques/quantization-pruning/python/quantization_pruning.py
Rscript techniques/quantization-pruning/r/quantization_pruning.R
```

**Refs:** Han, S., Mao, H. & Dally, W.J. "Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding." *ICLR*, 2016; Frankle, J. & Carbin, M. "The lottery ticket hypothesis: finding sparse, trainable neural networks." *ICLR*, 2019; Dettmers, T. et al. "LLM.int8(): 8-bit matrix multiplication for transformers at scale." *NeurIPS*, 2022; Frantar, E. et al. "GPTQ: accurate post-training quantization for generative pre-trained transformers." *ICLR*, 2023.

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
