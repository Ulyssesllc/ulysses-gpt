#!/usr/bin/env bash

# ==============================================================================
# check_code.sh - Quality Assurance Script for mini-gpt-scratch
# Performs Linter Check (Ruff), Type Checking (Mypy), and Unit Tests (Pytest).
# ==============================================================================

set -euo pipefail

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔍 [1/3] Running Ruff Linter & Formatter...${NC}"
if command -v ruff &> /dev/null; then
    ruff check .
    ruff format --check .
    echo -e "${GREEN}✓ Ruff checks passed!${NC}"
else
    echo -e "${RED}⚠️ Ruff not found. Run 'pip install -e \".[dev]\"' to install.${NC}"
fi

echo -e "\n${BLUE}🔬 [2/3] Running Mypy Static Type Checker...${NC}"
if command -v mypy &> /dev/null; then
    mypy src/
    echo -e "${GREEN}✓ Mypy type check passed!${NC}"
else
    echo -e "${RED}⚠️ Mypy not found. Run 'pip install -e \".[dev]\"' to install.${NC}"
fi

echo -e "\n${BLUE}🧪 [3/3] Running Pytest Unit Tests...${NC}"
if command -v pytest &> /dev/null; then
    pytest tests/ -v --cov=src --cov-report=term-missing
    echo -e "${GREEN}✓ All Unit Tests passed!${NC}"
else
    echo -e "${RED}⚠️ Pytest not found. Run 'pip install -e \".[dev]\"' to install.${NC}"
fi

echo -e "\n${GREEN}✅ Code quality verification complete!${NC}"
