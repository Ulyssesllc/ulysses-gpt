#!/usr/bin/env bash

# ==============================================================================
# run_project.sh - End-to-End Pipeline Execution Script
# Runs code verification first, then executes Pre-training, SFT, DPO & Inference.
# ==============================================================================

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE} Step 1: Running Code Quality Checks (check_code.sh)...${NC}\n"
if [ -f "./check_code.sh" ]; then
    bash ./check_code.sh
else
    echo -e "${RED} check_code.sh not found. Proceeding directly to training...${NC}"
fi

echo -e "\n${BLUE} Step 2: Executing Full LLM Pipeline (src/train.py)...${NC}"
echo -e "${BLUE}Pipeline: Data Encoding ➔ Pre-training ➔ SFT ➔ DPO ➔ Text Generation${NC}\n"

python3 train.py

echo -e "\n${GREEN} Project execution finished successfully!${NC}"
