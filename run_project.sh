#!/usr/bin/env bash

# End-to-end ulysses-gpt training pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
mkdir -p "$CHECKPOINT_DIR"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash run_project.sh

Runs pretraining, then SFT and DPO when their Hugging Face dataset variables are set.
The Hugging Face model directory is written to CHECKPOINT_DIR/final_model_path.txt.

Main environment variables:
  PYTHON_BIN                 Python executable (default: python3)
  CHECKPOINT_DIR             Output directory (default: ./checkpoints)
  RUN_CHECKS                 Set to 1 to run check_code.sh before training
  PRETRAIN_DATASET           Local UTF-8 text file (mutually exclusive with URL/HF)
  PRETRAIN_DATASET_URL       URL of a UTF-8 text file
  PRETRAIN_HF_DATASET        Hugging Face dataset ID; blank uses Tiny Shakespeare
  PRETRAIN_HF_CONFIG         Optional Hugging Face dataset config
  PRETRAIN_HF_SPLIT          Training split (default: train)
  PRETRAIN_VALIDATION_SPLIT  Validation split (default: validation)
  TEXT_COLUMN                Text column or auto (default: auto)
  MAX_TOKENS                 Maximum train tokens (default: 2000000)
  MAX_VAL_TOKENS             Maximum validation tokens (default: 20000)
  BLOCK_SIZE, BATCH_SIZE, GRAD_ACCUM_STEPS, EPOCHS, MAX_STEPS
  LEARNING_RATE, N_LAYER, N_HEAD, N_EMBD, SEED, PROMPT
  RESUME_FROM                Optional pretraining checkpoint to resume

Optional SFT (runs when SFT_HF_DATASET is non-empty):
  SFT_HF_DATASET, SFT_HF_CONFIG, SFT_HF_SPLIT
  SFT_PROMPT_COLUMN, SFT_CONTEXT_COLUMN, RESPONSE_COLUMN
  SFT_EPOCHS, SFT_MAX_SAMPLES, SFT_LR

Optional DPO (runs when DPO_HF_DATASET is non-empty):
  DPO_HF_DATASET, DPO_HF_CONFIG, DPO_HF_SPLIT
  CHOSEN_COLUMN, REJECTED_COLUMN, DPO_EPOCHS, DPO_MAX_SAMPLES, DPO_LR

SFT starts from ulysses-gpt-pretrain.pt. DPO starts from ulysses-gpt-sft.pt when
SFT ran, otherwise it starts from ulysses-gpt-pretrain.pt. The last enabled stage
is the final model.
Hugging Face export runs once, after the last enabled stage completes.
HELP
    exit 0
fi
if (( $# > 0 )); then
    echo "This script reads configuration from environment variables. Use --help for options." >&2
    exit 2
fi

if [[ "${RUN_CHECKS:-0}" == "1" ]]; then
    echo -e "${BLUE}Running code checks before training...${NC}"
    bash "$ROOT/check_code.sh"
fi

source_count=0
[[ -n "${PRETRAIN_DATASET:-}" ]] && ((source_count += 1))
[[ -n "${PRETRAIN_DATASET_URL:-}" ]] && ((source_count += 1))
[[ -n "${PRETRAIN_HF_DATASET:-}" ]] && ((source_count += 1))
if (( source_count > 1 )); then
    echo "Set only one of PRETRAIN_DATASET, PRETRAIN_DATASET_URL, PRETRAIN_HF_DATASET." >&2
    exit 2
fi

: "${BLOCK_SIZE:=128}"
: "${BATCH_SIZE:=8}"
: "${GRAD_ACCUM_STEPS:=2}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=2000}"
: "${MAX_TOKENS:=2000000}"
: "${MAX_VAL_TOKENS:=20000}"
: "${LEARNING_RATE:=0.0003}"
: "${N_LAYER:=4}"
: "${N_HEAD:=4}"
: "${N_EMBD:=256}"
: "${SEED:=42}"
: "${PROMPT:=Hello}"
: "${TEXT_COLUMN:=auto}"
: "${PRETRAIN_HF_SPLIT:=train}"
: "${PRETRAIN_VALIDATION_SPLIT:=validation}"

pretrain_args=(
    --task pretrain
    --block-size "$BLOCK_SIZE" --batch-size "$BATCH_SIZE"
    --grad-accum-steps "$GRAD_ACCUM_STEPS" --epochs "$EPOCHS"
    --max-steps "$MAX_STEPS" --max-tokens "$MAX_TOKENS"
    --max-val-tokens "$MAX_VAL_TOKENS" --learning-rate "$LEARNING_RATE"
    --n-layer "$N_LAYER" --n-head "$N_HEAD" --n-embd "$N_EMBD"
    --checkpoint-dir "$CHECKPOINT_DIR" --seed "$SEED" --prompt "$PROMPT"
)
if [[ -n "${PRETRAIN_DATASET:-}" ]]; then
    pretrain_args+=(--dataset "$PRETRAIN_DATASET")
elif [[ -n "${PRETRAIN_DATASET_URL:-}" ]]; then
    pretrain_args+=(--dataset-url "$PRETRAIN_DATASET_URL")
elif [[ -n "${PRETRAIN_HF_DATASET:-}" ]]; then
    pretrain_args+=(
        --hf-dataset "$PRETRAIN_HF_DATASET"
        --hf-split "$PRETRAIN_HF_SPLIT"
        --validation-split "$PRETRAIN_VALIDATION_SPLIT"
        --text-column "$TEXT_COLUMN"
    )
    [[ -n "${PRETRAIN_HF_CONFIG:-}" ]] && pretrain_args+=(--hf-config "$PRETRAIN_HF_CONFIG")
fi
[[ -n "${RESUME_FROM:-}" ]] && pretrain_args+=(--resume "$RESUME_FROM")

echo -e "${BLUE}Stage 1/3: pretraining${NC}"
    "$PYTHON_BIN" "$ROOT/train.py" "${pretrain_args[@]}" 2>&1
PRETRAIN_CHECKPOINT="$CHECKPOINT_DIR/ulysses-gpt-pretrain.pt"
[[ -f "$PRETRAIN_CHECKPOINT" ]] || { echo "Pretraining checkpoint missing: $PRETRAIN_CHECKPOINT" >&2; exit 1; }
FINAL_MODEL="$PRETRAIN_CHECKPOINT"

if [[ -n "${SFT_HF_DATASET:-}" ]]; then
    : "${SFT_HF_SPLIT:=train}"
    : "${SFT_PROMPT_COLUMN:=${PROMPT_COLUMN:-prompt}}"
    : "${SFT_CONTEXT_COLUMN:=context}"
    : "${RESPONSE_COLUMN:=response}"
    : "${SFT_EPOCHS:=1}"
    : "${SFT_MAX_SAMPLES:=10000}"
    : "${SFT_LR:=0.0001}"
    sft_args=(
        --task sft --hf-dataset "$SFT_HF_DATASET" --hf-split "$SFT_HF_SPLIT"
        --prompt-column "$SFT_PROMPT_COLUMN" --context-column "$SFT_CONTEXT_COLUMN"
        --response-column "$RESPONSE_COLUMN"
        --checkpoint "$PRETRAIN_CHECKPOINT" --checkpoint-dir "$CHECKPOINT_DIR"
        --block-size "$BLOCK_SIZE" --batch-size "$BATCH_SIZE"
        --epochs "$SFT_EPOCHS" --max-samples "$SFT_MAX_SAMPLES"
        --learning-rate "$SFT_LR" --seed "$SEED" --prompt "$PROMPT"
    )
    [[ -n "${SFT_HF_CONFIG:-}" ]] && sft_args+=(--hf-config "$SFT_HF_CONFIG")
    echo -e "${BLUE}Stage 2/3: supervised fine-tuning (SFT)${NC}"
    "$PYTHON_BIN" "$ROOT/train.py" "${sft_args[@]}" 2>&1
    SFT_CHECKPOINT="$CHECKPOINT_DIR/ulysses-gpt-sft.pt"
    [[ -f "$SFT_CHECKPOINT" ]] || { echo "SFT checkpoint missing: $SFT_CHECKPOINT" >&2; exit 1; }
    FINAL_MODEL="$SFT_CHECKPOINT"
fi

if [[ -n "${DPO_HF_DATASET:-}" ]]; then
    : "${DPO_HF_SPLIT:=train}"
    : "${PROMPT_COLUMN:=prompt}"
    : "${CHOSEN_COLUMN:=chosen}"
    : "${REJECTED_COLUMN:=rejected}"
    : "${DPO_EPOCHS:=1}"
    : "${DPO_MAX_SAMPLES:=10000}"
    : "${DPO_LR:=0.000005}"
    DPO_INIT_CHECKPOINT="$FINAL_MODEL"
    dpo_args=(
        --task dpo --hf-dataset "$DPO_HF_DATASET" --hf-split "$DPO_HF_SPLIT"
        --prompt-column "$PROMPT_COLUMN" --chosen-column "$CHOSEN_COLUMN"
        --rejected-column "$REJECTED_COLUMN" --checkpoint "$DPO_INIT_CHECKPOINT"
        --checkpoint-dir "$CHECKPOINT_DIR" --block-size "$BLOCK_SIZE"
        --batch-size "$BATCH_SIZE" --epochs "$DPO_EPOCHS"
        --max-samples "$DPO_MAX_SAMPLES" --learning-rate "$DPO_LR"
        --seed "$SEED" --prompt "$PROMPT"
    )
    [[ -n "${DPO_HF_CONFIG:-}" ]] && dpo_args+=(--hf-config "$DPO_HF_CONFIG")
    echo -e "${BLUE}Stage 3/3: Direct Preference Optimization (DPO)${NC}"
    "$PYTHON_BIN" "$ROOT/train.py" "${dpo_args[@]}" 2>&1
    DPO_CHECKPOINT="$CHECKPOINT_DIR/ulysses-gpt-dpo.pt"
    [[ -f "$DPO_CHECKPOINT" ]] || { echo "DPO checkpoint missing: $DPO_CHECKPOINT" >&2; exit 1; }
    FINAL_MODEL="$DPO_CHECKPOINT"
fi

HF_MODEL_DIR="$CHECKPOINT_DIR/ulysses-gpt"
echo -e "${BLUE}Exporting the final training checkpoint to Hugging Face format...${NC}"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m mini_gpt.hf_export \
    --checkpoint "$FINAL_MODEL" \
    --output-dir "$HF_MODEL_DIR" \
    --model-name "ulysses-gpt" 2>&1
printf '%s\n' "$HF_MODEL_DIR" > "$CHECKPOINT_DIR/final_model_path.txt"
echo -e "${GREEN}ulysses-gpt pipeline complete. Hugging Face model: $HF_MODEL_DIR${NC}"
