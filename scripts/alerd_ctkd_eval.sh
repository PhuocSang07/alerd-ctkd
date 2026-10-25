#!/bin/bash
# Evaluate a Hugging Face checkpoint on GSM8K test (1,319 problems) and MATH500 with the sgl evaluator, under the ALERD
# protocol: n=3 samples, T=0.6, top-p 0.9, repetition penalty 1.05, Math-Verify, pass@1 and pass@3. Needs a GPU with vLLM.
# run_sgl_eval.py adds GSM8K to sgl and builds the prompts like the training data (chat template when the model has
# one, raw instruction for GPT-2); sgl itself is only imported.
#
# Usage: SGL_SRC=/path/to/SpectralGuidedLearning/src bash scripts/alerd_ctkd_eval.sh <checkpoint> <tag> <base_model>
#   checkpoint: a trained output directory, a base model (zero-shot row) or the teacher (teacher row)
#   tag:        results land in ${RESULTS_DIR:-results}/<tag>/summary.json, which collect_results.py reads
#   base_model: supplies the tokenizer and prompt format (the student's, even when evaluating a checkpoint)
# MAX_TOKENS / MAX_MODEL_LEN default to 512 / 1024 for GPT-2 (1,024-token context) and 1024 / 2048 for other models.
set -euo pipefail

: "${SGL_SRC:?set SGL_SRC to the SpectralGuidedLearning src directory}"
CHECKPOINT=${1:?checkpoint}
TAG=${2:?tag}
BASE_MODEL=${3:?base model}
export PYTHONPATH="${SGL_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${BASE_MODEL}" == *gpt2* ]]; then
  DEFAULT_TOKENS=512; DEFAULT_LEN=1024
else
  DEFAULT_TOKENS=1024; DEFAULT_LEN=2048
fi

"${PYTHON:-python}" run_sgl_eval.py --model "${CHECKPOINT}" --tag "${TAG}" --base-model "${BASE_MODEL}" \
  --benchmarks gsm8k,math500 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 \
  --n-samples 3 --max-tokens "${MAX_TOKENS:-${DEFAULT_TOKENS}}" --max-model-len "${MAX_MODEL_LEN:-${DEFAULT_LEN}}" \
  --batch-size 64 --gpu-memory-utilization 0.8 --enforce-eager --seed "${EVAL_SEED:-42}" \
  --grader math_verify --results-dir "${RESULTS_DIR:-results}"
