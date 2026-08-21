# Contrastive Representation Learning (Reference §27.x extra)

Learn embeddings without labels by pulling **positive pairs** (two augmented
views of the same input) together and pushing **negative pairs** (views of
different inputs) apart.

## SimCLR NT-Xent loss (Chen et al. 2020)

For a batch of `N` inputs, generate two views per input; encode both with
the same encoder + projection head; L2-normalise:

```
z_i = g(f(x_i)) / ‖g(f(x_i))‖
L_i = − log( exp(z_i · z'_i / τ) / Σ_{k ≠ i} exp(z_i · z_k / τ + z_i · z'_k / τ) )
```

- `τ ≈ 0.1–0.5` — softmax temperature; smaller = sharper contrastive signal.
- Requires large batch (or momentum queue) for enough negatives.
- Projector `g` (usually a 2-layer MLP) is discarded after pretraining; downstream tasks use `f` only.

## Modern family

| Method | Key idea |
|---|---|
| **SimCLR** (Chen 2020) | large-batch NT-Xent + projector |
| **MoCo** (He 2020) | momentum encoder + queue of negatives (small batch OK) |
| **BYOL** (Grill 2020) | no negatives; target network + predictor + EMA |
| **SimSiam** (Chen 2021) | BYOL without target network; stop-gradient trick |
| **DINO** (Caron 2021) | teacher-student self-distillation; sharp / centring |
| **CLIP** (Radford 2021) | image-text contrastive; bi-encoder trained on 400M pairs |
| **SimCSE** (Gao 2021) | contrastive sentence embeddings |
| **BEiT / MAE / IJEPA** | masked-image variants; reconstructive rather than contrastive |

## When to use

- **Pretraining an encoder** on unlabelled data before a small labelled downstream fine-tune.
- **Bi-encoder retrieval** (CLIP, sentence-transformers, E5 / GTE / BGE for text).
- **Metric learning** — face verification, speaker verification.
- **Recommendation** — user × item bi-encoder.
- **Domain adaptation** — pretrain contrastively on unlabelled target-domain data.

## Files

- `python/contrastive_learning.py` — from-scratch SimCLR-style NT-Xent with a linear encoder + L2 normalisation + numerical gradient (kept short for a numpy demo). Toy setup: 15 latent identities in ℝ⁶, each with two noisy views; train an encoder to align matched views. After 40 epochs: positive-pair cosine +0.982, negative-pair −0.053, 93.3% at rank-1 retrieval (mean rank 1.07).
- `r/contrastive_learning.R` — `torch` (manual NT-Xent); Python `lightly`, `pytorch-metric-learning`, `sentence-transformers`, `openclip`.

## Assumptions & caveats

- **Augmentation is the design decision** — SimCLR's success comes from strong image augmentations (crop, colour jitter, blur); weak augmentations underperform.
- **Batch size / negatives** — SimCLR needs 4k–8k batches; MoCo / BYOL / SimSiam remove this constraint.
- **Projection head matters** — remove it for downstream; keeping it hurts.
- **Alignment vs uniformity** (Wang-Isola 2020) — good contrastive embeddings simultaneously align positives and uniformly cover the sphere.
- **Collapse** — trivial constant embeddings satisfy the alignment term; negatives / target networks / centring prevent this.
- **Cross-modal** contrastive (CLIP) is what enables text-to-image search, zero-shot classification, embedding-based RAG.
- **Fair comparison to supervised** — contrastive pretraining + linear probe typically reaches ~90% of supervised accuracy on ImageNet at the same architecture; with fine-tuning, matches or exceeds.

## Related in this repo

- `transfer-learning` — the downstream fine-tune after contrastive pretraining.
- `sentence-similarity`, `word-embeddings`, `graph-neural-network` — embedding models that benefit from contrastive pretraining.
- `variational-autoencoder`, `autoencoder`, `gan-training` — alternative unsupervised representation-learning families.

## Run

```
python techniques/contrastive-learning/python/contrastive_learning.py
Rscript techniques/contrastive-learning/r/contrastive_learning.R
```

**Refs:** Chen, T. et al. "A simple framework for contrastive learning of visual representations (SimCLR)." *ICML*, 2020; He, K. et al. "Momentum contrast for unsupervised visual representation learning (MoCo)." *CVPR*, 2020; Grill, J.-B. et al. "Bootstrap Your Own Latent (BYOL)." *NeurIPS*, 2020; Radford, A. et al. "Learning transferable visual models from natural language supervision (CLIP)." *ICML*, 2021.

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
