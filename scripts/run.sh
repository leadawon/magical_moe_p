#!/bin/bash
# MoE routing probe — full run on GPUs 0,1,2,3
set -e

VENV="/data1/ai25170474/workspace/venvs/qwenmoevenv"
PYTHON="$VENV/bin/python3.10"
PROJECT="/data1/ai25170474/workspace/magical_moe_probe"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"

NUM_EX="${1:-200}"

echo "============================================================"
echo "MoE ROUTING PROBE"
echo "Model: Qwen3-30B-A3B   GPUs: 0,1,2,3"
echo "Datasets: gsm8k svamp humaneval_plus mbpp_plus mnli snli"
echo "Examples/dataset: $NUM_EX   (forward-only, no generation)"
echo "============================================================"

# 1) capture routing
$PYTHON -u -m src.probe --num_examples "$NUM_EX"

# 2) analyse + test hypotheses
$PYTHON -u -m src.analyze

echo "============================================================"
echo "DONE. See results/FINDINGS.md"
echo "============================================================"
