# 🚀 `ulysses-gpt`

> **A Production-Ready, End-to-End Generative Transformer Built from Scratch in PyTorch**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![CI/CD: GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)](https://github.com/)
[![Docker: Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`mini-gpt-scratch` is a complete, optimized, and mathematically rigorous implementation of an autoregressive causal language model (GPT architecture), built entirely from scratch with PyTorch.

The project is designed to bridge the gap between foundational deep learning theory (**Stanford CS224N**, **LLMs from Scratch**) and production software engineering practices (**Software Engineering & DevOps**). It covers the complete lifecycle of a modern LLM: **Subword BPE Tokenization ➔ Pre-training (AMP FP16 & Grad Accumulation) ➔ Autoregressive Inference (Temperature Scaling + Top-k) ➔ SFT Instruction Tuning ➔ DPO Preference Alignment ➔ CI/CD & Docker Containerization**.

---

## 🔑 Core Features

- **Causal Transformer Architecture (Decoder-Only)**:
  - Multi-Head Causal Self-Attention with causal masking (lower-triangular mask), enforcing the $O(N^2)$ causal attention principle.
  - Pre-Layer Normalization and Residual Connections for stable gradient flow.
  - Weight tying between the token embedding and LM output head to reduce the parameter count.
- **Subword BPE Tokenizer (`tiktoken`)**:
  - Integrated GPT-2 BPE encoder (`vocab_size = 50,257`), providing 2.5x–3x higher text compression density than a character tokenizer.
- **Resource-Efficient Pre-training Engine**:
  - **Automatic Mixed Precision (AMP FP16)** reduces GPU VRAM usage by 50%.
  - **Gradient Accumulation** simulates a larger effective batch size on a T4 GPU (Google Colab / Kaggle).
  - **Cosine Annealing Learning Rate Scheduler** with linear warmup.
  - **Automatic Checkpointing** saves and restores the state of the `model`, `optimizer`, `scaler`, and `global_step`.
  - Automatic Perplexity evaluation ($PPL = e^{\text{Loss}}$) and validation tracking.
- **Autoregressive Text Generation Engine**:
  - Autoregressive sampling with integrated **Temperature Scaling** and **Top-k Filtering**.
- **Post-Training & Alignment**:
  - **Supervised Fine-Tuning (SFT)** with Prompt Masking (`-100` target loss ignoring).
  - **Direct Preference Optimization (DPO)** using a dual-model architecture (Policy Model vs. Frozen Reference Model) with Reward Margin tracking.
- **Software Engineering & CI/CD Workflow**:
  - Standard PEP 621 package declaration (`pyproject.toml`).
  - Strict static type checking (`mypy`), linting/formatting (`ruff`), and 100% passing unit tests (`pytest`).
  - Container packaging with **Dockerfile** and automated shell scripts (`check_code.sh`, `run_project.sh`).
  - **GitHub Actions CI/CD** workflow for automated testing.

---

## 📐 Theory & Mathematical Foundations

### 1. Causal Self-Attention

For an input token sequence $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$, Queries ($Q$), Keys ($K$), and Values ($V$) are calculated through linear projections:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Here, $M_{i,j} = 0$ when $i \ge j$ and $-\infty$ when $i < j$ (the lower-triangular mask).

### 2. Autoregressive Pre-Training Loss

Cross-Entropy Loss evaluates next-token prediction:

$$\mathcal{L}_{\text{Pretrain}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$

### 3. Direct Preference Optimization (DPO) Loss

DPO directly aligns model preferences using chosen ($y_w$) and rejected ($y_l$) response pairs under the Bradley-Terry model, without requiring a separate reward model:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

## 📂 Project Structure

```text
ulysses-gpt/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD workflow
├── pyproject.toml             # PEP 621, Ruff, Mypy, and Pytest configuration
├── Dockerfile                 # Docker container packaging
├── check_code.sh              # Fail-fast code quality check script
├── run_project.sh             # Script for running the complete project pipeline
├── README.md                  # Detailed project documentation
├── src/
│   ├── __init__.py            # Main exports (MiniGPT, GPTConfig, tokenizers, etc.)
│   ├── model.py               # Causal Self-Attention, MLP, and MiniGPT architecture
│   ├── dataset.py             # BPETokenizer (tiktoken) and sliding-window dataset
│   ├── trainer.py             # Pre-training engine (AMP FP16, Cosine LR, checkpointing)
│   ├── generate.py            # Autoregressive sampling (Temperature, Top-k)
│   ├── post_train.py          # SFT masking and DPO alignment engine
│   └── train_colab.py         # End-to-end training pipeline for Colab
└── tests/
    ├── __init__.py
    ├── test_model.py          # Tensor shape, masking, and loss unit tests
    ├── test_trainer.py        # Trainer loop and checkpointing unit tests
    └── test_post_train.py     # SFT masking, log-probability, and DPO loss unit tests
```

---

## ⚙️ Default Model Configuration (`GPTConfig`)

| Hyperparameter | Default Value | Description |
| :--- | :--- | :--- |
| **Parameters** | `~3.5M` – `15M` | Suitable for fast training on a Colab T4 GPU |
| **Embedding Dim (`n_embd`)** | `256` / `384` | Transformer hidden dimension |
| **Attention Heads (`n_head`)** | `4` / `6` | Number of attention heads (`head_dim = n_embd / n_head`) |
| **Layers (`n_layer`)** | `4` / `6` | Number of Transformer decoder blocks |
| **Context Length (`block_size`)** | `128` / `256` | Maximum context length |
| **Vocab Size (`vocab_size`)** | `50,257` | GPT-2 standard BPE vocabulary size |
| **Optimizer** | `AdamW` | `lr=1e-3, beta1=0.9, beta2=0.95, weight_decay=0.1` |

---

## 🛠️ Installation & Usage

### 1. Local Setup

```bash
# Clone the repository
!git clone https://github.com/Ulyssesllc/ulysses-gpt.git
%cd ulysses-gpt

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Code Quality Checks & Unit Tests

Grant execute permission and run the fail-fast validation script:

```bash
chmod +x check_code.sh
./check_code.sh
```

### 3. Run the Local Training Pipeline

```bash
chmod +x run_project.sh
./run_project.sh
```

### 4. Run the End-to-End Demo & Chatbot UI on Google Colab

Open a new **Google Colab** notebook, select **Runtime ➔ T4 GPU**, and run:

```python
!git clone https://github.com/Ulyssesllc/ulysses-gpt.git
%cd ulysses-gpt
!pip install -e ".[dev]" gradio
!python train.py
```

---

## 🌐 Custom Training Datasets

### Choose a pre-training text dataset

`train.py` accepts a local UTF-8 `.txt` corpus or a direct URL to a UTF-8 text file.
With no option it downloads and uses Tiny Shakespeare, as before.

```bash
# Use a local text corpus
python train.py --dataset data/my_corpus.txt

# Download a text file (saved under data/ using the URL filename)
python train.py --dataset-url https://example.org/my_corpus.txt
```

You can also load a dataset from the Hugging Face Hub. `--text-column` defaults
to automatic detection of common text fields (`text`, `content`, `article`,
`messages`, and others); pass a column name to override it. Rows are normalized
and encoded in batches, with visible progress and a configurable token limit.

```bash
python train.py --hf-dataset Salesforce/wikitext --hf-config wikitext-2-raw-v1 \
  --hf-split train --text-column text --min-text-characters 20
```

The same cleaning rules are applied to local and downloaded text files.

The Hugging Face `validation` split is used when available; otherwise 5% of
the training rows are held out deterministically. `--max-tokens` bounds the
tokenized corpus retained in memory, and `--max-steps` bounds optimizer updates.
Model size, batch size, context length, learning rate, epochs, seed, and
checkpoint paths can be set from the command line. Resume a pretraining run
with `--resume checkpoints/final_minigpt.pt`.

SFT and DPO now read real Hugging Face columns and require a pretrained
checkpoint. For example:

```bash
python train.py --task sft --hf-dataset org/instructions \
  --prompt-column prompt --response-column response \
  --checkpoint checkpoints/final_minigpt.pt

python train.py --task dpo --hf-dataset org/preferences \
  --prompt-column prompt --chosen-column chosen --rejected-column rejected \
  --checkpoint checkpoints/final_minigpt.pt
```

`--max-samples` limits the SFT/DPO examples loaded into memory. These tasks
expect their respective columns to contain text or common nested conversation
structures; dataset-specific schemas can be mapped with the column options.

---

## 📚 References & Acknowledgments

The `ulysses-gpt` project was built by synthesizing academic and software engineering resources from the following sources:

1. **Stanford CS224N: Natural Language Processing with Deep Learning** (Prof. Christopher Manning, Prof. Diyi Yang, Shikhar Murty)
   - *Materials*: Lecture slides, syllabus, and the MinGPT Default Final Project.
   - *Contribution*: Theoretical foundations for Causal Self-Attention, the Transformer decoder, GPT-2 architecture, SFT instruction tuning, and Direct Preference Optimization (DPO).
2. **Build a Large Language Model (From Scratch)** (Sebastian Raschka, Manning Publications)
   - *Materials*: Book content and the `rasbt/LLMs-from-scratch` GitHub repository.
   - *Contribution*: A methodology for implementing a GPT model step by step with pure PyTorch, integrating a BPE tokenizer (`tiktoken`) and a training loop structure.

---

## 📜 License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
