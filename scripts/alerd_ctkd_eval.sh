#!/bin/bash
# Evaluate a Hugging Face checkpoint with the sgl evaluator under the ALERD / P-ALIGN protocol
# (sgl configs/common/eval/palign-protocol.yaml: thinking off, n=3, T=0.6, top-p 0.9, repetition penalty 1.05,
# 4096 tokens, AIME25 / AIME24 / AMC12 / MATH500, graded by Math-Verify). Needs a GPU with vLLM.
#
# Usage: SGL_SRC=/path/to/SpectralGuidedLearning/src bash scripts/alerd_ctkd_eval.sh <checkpoint> <tag> <base_model>
#   checkpoint: a trained output directory, a base model (zero-shot row) or the teacher (teacher row)
#   tag:        results land in ${RESULTS_DIR:-results}/<tag>/summary.json, which collect_results.py reads
#   base_model: supplies the tokenizer and chat template (the student's, even when evaluating a checkpoint)
set -euo pipefail

: "${SGL_SRC:?set SGL_SRC to the SpectralGuidedLearning src directory}"
CHECKPOINT=${1:?checkpoint}
TAG=${2:?tag}
BASE_MODEL=${3:?base model}
export PYTHONPATH="${SGL_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

python -m sgl.eval.evaluate --model "${CHECKPOINT}" --tag "${TAG}" --base-model "${BASE_MODEL}" \
  --benchmarks math500,aime24,aime25,amc12 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 \
  --n-samples 3 --max-tokens 4096 --max-model-len 4096 --batch-size 64 --gpu-memory-utilization 0.8 \
  --enforce-eager --seed "${EVAL_SEED:-42}" --chat-template --no-enable-thinking --palign-prompt \
  --grader math_verify --results-dir "${RESULTS_DIR:-results}"
