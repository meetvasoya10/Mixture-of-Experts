# Mixture of Experts (MoE) - Custom Implementation

This repository provides a clean and extensible implementation of a **Mixture of Experts (MoE)** model, where inputs are routed to specialized expert networks via a learned gating mechanism. The implementation integrates seamlessly with GPT-Neo and supports modular pretraining, fine-tuning, and inference.

---

## What is Mixture of Experts (MoE)?

A **Mixture of Experts** model is a neural architecture where multiple expert subnetworks ("experts") specialize in different data patterns or tasks. A **gating network** learns to assign weights to each expert based on the input, allowing only a few experts to be active per input instance. This leads to:

- Improved **efficiency** through sparse activation.
- Better **generalization** by leveraging specialization.
- Scalability for multi-domain or multi-task learning.

In this implementation:
- Each expert is a feed-forward neural network.
- The gating network computes a softmax distribution to route tokens to experts.
- MoE is integrated into GPT-Neo’s architecture by replacing its MLP block with a custom MoE layer.

---

## Project Structure

```
MoE/
├── data/                       # Preprocessed datasets (placeholder .gitkeep included)
│   ├── coding/
│   ├── math/
│   └── cot/

├── datasets/                  # Saved HuggingFace datasets (ignored in Git, .gitkeep added)
│   └── .gitkeep

├── scripts/                    # Dataset preparation scripts
│   ├── prepare_coding_dataset.py
│   ├── prepare_math_dataset.py
│   └── prepare_cot_dataset.py

├── moe/                        # MoE model and expert logic
│   ├── moe.py                  # MoE integration into transformer
│   ├── gating.py              # Gating network
│   ├── experts.py             # Expert network definitions
│   └── examples/              # Training and inference scripts
│       ├── train_moe.py       # Pretrain experts independently
│       ├── fine_tune_moe.py   # Integrate MoE into GPT-Neo and fine-tune
│       ├── infer_coding.py    # Run inference using coding expert
│       └── checkpoints/       # Saved experts and fine-tuned model (.gitkeep added)

├── requirements.txt           # Required Python packages
├── .gitignore
└── README.md
```

> **Note**: Folders like `data/`, `datasets/`, and `checkpoints/` are ignored by Git and preserved using `.gitkeep` files. These are populated dynamically during training or data preparation.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/moe-project.git
cd moe-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare datasets

```bash
python scripts/prepare_coding_dataset.py
python scripts/prepare_cot_dataset.py
python scripts/prepare_math_dataset.py
```

### 4. Pretrain experts

```bash
python moe/examples/train_moe.py --data_path ./datasets
```

### 5. Fine-tune MoE integrated with GPT-Neo

```bash
python moe/examples/fine_tune_moe.py --data_path ./datasets
```

### 6. Run inference with a coding expert

```bash
python moe/examples/infer_coding.py --checkpoint_path ./moe/examples/checkpoints/expert_coding_pretrained.pt
```

---

## Key Components

- `experts.py` – Feed-forward expert networks with GELU activations.
- `gating.py` – Gating mechanism using a linear layer + softmax.
- `moe.py` – MoE wrapper that replaces GPT-Neo’s MLP block.
- `train_moe.py` – Pretrains each expert using AMP and token-level supervision.
- `fine_tune_moe.py` – Freezes GPT-Neo weights and trains only MoE block.
- `infer_coding.py` – Performs generation using a single coding expert.

---

## Training Tips

- **Batch Size**: Start with 2–8 depending on sequence length and GPU memory. Gradually scale if memory allows.
- **Epochs**: 5–50 depending on task complexity and convergence speed.
- **Optimizer**: AdamW is used with a learning rate of 1e-5. You may try schedulers like cosine annealing for stability.
- **Precision**: Use mixed-precision training (`torch.cuda.amp`) to reduce memory usage and speed up training.
- **Hardware**: A CUDA-enabled GPU with ≥12GB memory is recommended for training with 2048-d hidden sizes.

---
