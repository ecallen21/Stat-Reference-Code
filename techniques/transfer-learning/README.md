# Transfer Learning (Reference §27.x extra)

Reuse a network pretrained on a large upstream task for a smaller downstream
task. Two dominant recipes:

## Feature extraction

- **Freeze** the backbone (encoder / body).
- Train only a **fresh head** on top.
- Cheap; low overfitting risk; needs the backbone to already contain useful features for the target task.

## Fine-tuning

- Start from the pretrained weights.
- Train the whole network at a **small learning rate** (or gradually unfreeze from the top).
- More expressive; needs more data than feature extraction; risks catastrophic forgetting on the source features.

## Advanced patterns

- **Discriminative LRs** — different LR per layer / block (higher for the head, lower deep in the backbone). ULMFiT and BERT fine-tuning.
- **Gradual unfreezing** — unfreeze top block first, train, unfreeze next, etc.
- **Adapters** (Houlsby 2019) — tiny bottleneck modules injected between frozen layers; only the adapters train.
- **LoRA** (Hu 2021) — low-rank update matrices; standard for cheap LLM fine-tuning.
- **Prompt-tuning / prefix-tuning** — train only a small learnable prompt / prefix.
- **P-tuning v2 / soft prompts** — extended prompt-tuning.

## When to use

- **Small labelled dataset** on a task where a related large-dataset pretrained model exists — always try transfer.
- **Compute-limited** — feature extraction on a frozen model is trivially cheap.
- **Rapid prototyping** — a frozen encoder + linear head gets you 80% of SOTA for free.
- **Domain-adaptation** — fine-tune the encoder on unlabelled target-domain text before the labelled downstream fine-tune.

## Files

- `python/transfer_learning.py` — from-scratch numpy MLP demo. Source task: 3-class 2-D Gaussians (n=1200, train acc 0.97). Target task: related 4-class task with only 12 labelled examples. Compares (a) train from scratch, (b) frozen backbone + fresh head, (c) unfrozen fine-tune at lr=0.005. On this low-dimensional toy task the three are within ~3% (fine-tune 0.777, scratch 0.761, feature 0.747); on high-dimensional real tasks (ImageNet-pretrained ResNet on CIFAR-10, BERT on GLUE) transfer typically wins by 10–30%.
- `r/transfer_learning.R` — `torchvision::model_resnet50(pretrained=TRUE)`, `keras3::application_resnet50(include_top=FALSE)`, `torch::optim_adam` with parameter groups; Python `torchvision.models`, `timm`, `huggingface transformers .from_pretrained`.

## Assumptions & caveats

- **Domain gap matters** — a photograph-pretrained model transfers poorly to satellite / medical imaging without domain-adaptive pretraining.
- **Head-only training with a bad backbone** can be worse than random init if the backbone's features aren't discriminative for the target.
- **Batch normalisation statistics** — freeze BN in `.eval()` mode when fine-tuning; using train-mode BN with a mini-batch of 4 target examples destroys the running stats.
- **Learning-rate scale** for fine-tuning is 10–100× smaller than for training from scratch; otherwise the pretrained features get overwritten.
- **Catastrophic forgetting** — plain fine-tuning erases upstream skills; freeze / adapters / LoRA / prompt-tuning preserve them.
- **Label mismatch between source and target** is fine because you replace the head; feature representations transfer.

## Related in this repo

- `deep-mlp-backprop`, `convolutional-nn`, `transformer-encoder` — architectures where transfer is standard.
- `contrastive-learning` — a way to pretrain without labels on the source.
- `lr-schedules`, `adam-optimizer` — the training-loop counterparts.
- `class-imbalance`, `calibration-scaling` — critical for the labelled downstream fine-tune when target labels are scarce.

## Run

```
python techniques/transfer-learning/python/transfer_learning.py
Rscript techniques/transfer-learning/r/transfer_learning.R
```

**Refs:** Yosinski, J. et al. "How transferable are features in deep neural networks?" *NeurIPS*, 2014; Howard, J. & Ruder, S. "Universal language model fine-tuning for text classification (ULMFiT)." *ACL*, 2018; Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

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
