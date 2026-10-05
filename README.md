# ulysses-gpt

A small decoder-only GPT implemented in PyTorch and exported as a Hugging Face-compatible GPT-2 model. The project covers text preprocessing, pretraining, text generation, supervised fine-tuning (SFT), and Direct Preference Optimization (DPO). The `run_project.sh` script can orchestrate these stages as one pipeline.

> This project produces a causal language model, not a ready-made assistant. Output quality depends on the training corpus, model size, and training time. Tiny Shakespeare is provided as a quick-start dataset.

## Features

- **Decoder-only Transformer:** causal self-attention, pre-layer normalization, residual connections, and tied token/output embeddings.
- **GPT-2 BPE tokenizer:** `tiktoken` encoding with a 50,257-token vocabulary.
- **Pretraining:** next-token prediction, local/URL/Hugging Face text sources, validation metrics, cosine learning-rate decay with warmup, gradient accumulation, gradient clipping, and CUDA automatic mixed precision.
- **Checkpointing:** saves model, optimizer, scaler, training step, and model configuration; supports resuming pretraining.
- **Hugging Face export:** after the final enabled training stage, writes `config.json`, `model.safetensors`, and GPT-2 tokenizer files to a `ulysses-gpt/` model directory. It loads through `AutoModelForCausalLM` and `AutoTokenizer`.
- **Generation:** autoregressive sampling with temperature and top-k filtering.
- **SFT:** trains on prompt/response pairs and masks prompt tokens from the response loss.
- **DPO:** preference alignment with a frozen reference model and chosen/rejected response pairs.
- **Engineering:** pytest suite, Ruff, Mypy, GitHub Actions, and Docker support.

## Requirements and installation

- Python 3.12 or later
- PyTorch 2.2 or later
- A CUDA GPU is recommended for training; CPU is supported but slower.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Install development tools when needed:

```bash
pip install -e ".[dev]"
```

## Run the full pipeline

Run from a Bash shell (Linux, macOS, WSL, Git Bash, or Google Colab):

```bash
bash run_project.sh
```

With no dataset settings, the script downloads Tiny Shakespeare and runs pretraining. SFT and DPO run only when their dataset IDs are configured. The script stops on errors and exports the final model to `CHECKPOINT_DIR/ulysses-gpt/` and writes that directory path to `CHECKPOINT_DIR/final_model_path.txt`.

To see all supported environment variables and defaults:

```bash
bash run_project.sh --help
```

### Pretraining data

Set at most one source. Hugging Face text columns are detected automatically by default; set `TEXT_COLUMN` to select one explicitly.

```bash
# Local UTF-8 text file
export PRETRAIN_DATASET="data/corpus.txt"

# Or a URL to a UTF-8 text file
# export PRETRAIN_DATASET_URL="https://example.org/corpus.txt"

# Or a Hugging Face dataset
# export PRETRAIN_HF_DATASET="Salesforce/wikitext"
# export PRETRAIN_HF_CONFIG="wikitext-2-raw-v1"
# export PRETRAIN_HF_SPLIT="train"
# export TEXT_COLUMN="text"

export CHECKPOINT_DIR="checkpoints"
export MAX_STEPS=2000
export MAX_TOKENS=2000000
export BLOCK_SIZE=128
export BATCH_SIZE=8
export GRAD_ACCUM_STEPS=2
export N_LAYER=4
export N_HEAD=4
export N_EMBD=256
bash run_project.sh
```

The default limits are intended as a starting point, not as a quality target. Increase training steps and use a representative corpus for a useful model. `N_EMBD` must be divisible by `N_HEAD`.

To resume pretraining from a checkpoint:

```bash
export RESUME_FROM="checkpoints/ulysses-gpt-pretrain.pt"
bash run_project.sh
```

Use the same model configuration and compatible training data when resuming.

### Optional SFT and DPO

SFT expects prompt and response columns. DPO expects prompt, chosen, and rejected columns. Each stage can use a separate Hugging Face dataset and config.

```bash
export SFT_HF_DATASET="your-org/instruction-data"
export SFT_PROMPT_COLUMN="prompt"
export SFT_CONTEXT_COLUMN="context" # optional; appended when non-empty
export RESPONSE_COLUMN="response"
export SFT_EPOCHS=1
export SFT_MAX_SAMPLES=2000
export SFT_LR=0.0001

export DPO_HF_DATASET="your-org/preference-data"
export PROMPT_COLUMN="prompt"
export CHOSEN_COLUMN="chosen"
export REJECTED_COLUMN="rejected"
export DPO_EPOCHS=1
export DPO_MAX_SAMPLES=2000
export DPO_LR=0.000005

bash run_project.sh
```

DPO starts from the SFT checkpoint when SFT is enabled; otherwise it starts from the pretrained checkpoint. Both stages are skipped when their dataset variables are empty. Dataset schemas vary, so set the column names to match your data.

### Pipeline settings

Common pretraining settings include `CHECKPOINT_DIR`, `BLOCK_SIZE`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `EPOCHS`, `MAX_STEPS`, `MAX_TOKENS`, `MAX_VAL_TOKENS`, `LEARNING_RATE`, `N_LAYER`, `N_HEAD`, `N_EMBD`, `SEED`, and `PROMPT`. Dataset and checkpoint options are listed by `bash run_project.sh --help`.

Set `RUN_CHECKS=1` to run Ruff, Mypy, and pytest before training. The default is `0` so training does not run the quality suite on every launch.


## Model artifacts

The pipeline keeps native training checkpoints (`ulysses-gpt-pretrain.pt`, `ulysses-gpt-sft.pt`, and `ulysses-gpt-dpo.pt`) for stage-to-stage training and resume. After all enabled stages finish, it exports only the final checkpoint to `CHECKPOINT_DIR/ulysses-gpt/` in Hugging Face format. The final directory contains `config.json`, `model.safetensors`, tokenizer files, and a model card. `final_model_path.txt` records this directory.

The export uses the standard Hugging Face GPT-2 architecture and tokenizer. Its `model_name` metadata is `ulysses-gpt`; its Transformers `model_type` remains `gpt2` so standard Transformers classes can load it without custom remote code.

## Load a trained model and generate text

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "checkpoints/ulysses-gpt"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

inputs = tokenizer("Hello", return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_k=40,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Native `.pt` training checkpoints serialize a Python `GPTConfig` object and should only be loaded with `weights_only=False` when trusted. For standard inference and sharing, use the Hugging Face directory shown above.

## Development checks

```bash
bash check_code.sh
```

This runs Ruff, Mypy, and pytest when those tools are installed. CI also runs these checks and builds the Docker image.

## Training and inference metrics

Each run appends JSON Lines records to `CHECKPOINT_DIR/metrics.jsonl`:

- Pretraining: training loss, validation cross-entropy, perplexity, and next-token accuracy.
- SFT: response-only masked loss, validation response loss, ROUGE-L, and exact match. The SFT dataset reserves 5% for validation when at least two examples are available; generation-based scores use at most `--eval-samples` examples (default 16).
- DPO: implicit loss, chosen/rejected reward, reward margin, and chosen-over-rejected win rate.
- Final sample generation: time to first token (TTFT), generation duration, and tokens per second.

ROUGE-L and exact match use whitespace-tokenized, lowercased/whitespace-normalized text. Exact match is therefore normalized exact match, not byte-for-byte equality. The repository does not currently include a Gradio UI or an external LLM judge; the inference timing helper can be reused by a serving UI, while judge scores require a separately configured evaluation provider.

## Project layout

```text
src/mini_gpt/
  model.py       Transformer model and configuration
  dataset.py     Text cleaning, tokenizers, and next-token dataset
  trainer.py     Pretraining loop, metrics, and checkpoint management
  generate.py    Autoregressive text generation
  hf_export.py   Hugging Face GPT-2 and tokenizer export
  post_train.py  SFT/DPO datasets, losses, and DPO trainer
train.py         Dataset loading and task entry point
run_project.sh   Environment-configured end-to-end pipeline
check_code.sh    Lint, type-check, and test runner
tests/           Unit tests
```

## License

MIT. See [LICENSE](LICENSE).
