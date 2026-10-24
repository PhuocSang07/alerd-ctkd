#!/bin/bash
# Per-student data preparation, answer-gain and predictability scoring (needs a GPU with vLLM; not meant for a laptop).
# Usage: SGL_SRC=/path/to/SpectralGuidedLearning/src bash scripts/score_gain.sh [WORK_DIR] [MAX_TOKENS]
# Authentication for gated models comes from the HF_TOKEN environment variable, never from this script.
set -euo pipefail

: "${SGL_SRC:?set SGL_SRC to the SpectralGuidedLearning src directory}"
WORK_DIR=${1:-work}
MAX_TOKENS=${2:-8192}
export PYTHONPATH="${SGL_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

STUDENTS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-8B"
)

mkdir -p "${WORK_DIR}"
python prepare_sra_data.py filter --output "${WORK_DIR}/s1k-boxed.jsonl"

for student in "${STUDENTS[@]}"; do
  tag=$(basename "${student}")
  out="${WORK_DIR}/${tag}"
  mkdir -p "${out}"
  python -m sgl.data.prepare --dataset-name "${WORK_DIR}/s1k-boxed.jsonl" --tokenizer "${student}" \
    --chat-template --no-enable-thinking --palign-prompt --max-tokens "${MAX_TOKENS}" \
    --output-path "${out}/train-segmented.jsonl"
  python -m sgl.signals.answer_gain --data-path "${out}/train-segmented.jsonl" --model "${student}" \
    --output "${out}/gain.json"
  python score_predictability.py --data "${out}/train-segmented.jsonl" --model "${student}" \
    --output "${out}/predictability.json"
  python prepare_sra_data.py convert --segmented "${out}/train-segmented.jsonl" \
    --student-tokenizer "${student}" --output "${out}/train-sra.jsonl"
done
