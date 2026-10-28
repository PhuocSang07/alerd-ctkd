#!/bin/bash
# Evaluates the checkpoints of scripts/span_kd_train.sh (and, with ZERO_SHOT=1, the student itself) with vLLM.
#
# Usage: [STUDENT=...] [OUT_ROOT=...] [RESULTS_DIR=...] [EXPERIMENTS="sft kd-token ..."] [ZERO_SHOT=1] [PROMPT_STYLE=tsdkd] bash scripts/span_kd_eval.sh
#
# Per model: GSM8K test and MATH500 with 3 samples per problem, AIME24 and AIME25 (30 problems each) with 16, at
# temperature 0.6, top-p 0.9 and MAX_TOKENS new tokens (4096). pass@1 is the mean over all samples, pass@3 the unbiased
# estimate (the plain "any of 3" for 3 samples), pass@16 = any correct. The summaries are RESULTS_DIR/<name>/<suite>/summary.json.
set -euo pipefail

STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
OUT_ROOT=${OUT_ROOT:-outputs/span-kd}
RESULTS_DIR=${RESULTS_DIR:-results/span-kd}
EXPERIMENTS=${EXPERIMENTS:-"sft kd-token gain-ce gain-kd gain-both"}
ZERO_SHOT=${ZERO_SHOT:-0}
PROMPT_STYLE=${PROMPT_STYLE:-tsdkd}  # the TSD-KD instruction of the training data; alerd for GSM8K-trained students
MAX_TOKENS=${MAX_TOKENS:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
PYTHON=${PYTHON:-python}

evaluate() {  # evaluate <name> <model path or hub id>
  local name=$1 model=$2
  for suite in "gsm8k,math500 3" "aime24,aime25 16"; do
    set -- ${suite}
    "${PYTHON}" benchmark_eval.py --model "${model}" --tag "${1%%,*}" --base-model "${STUDENT}" --benchmarks "$1" \
      --prompt-style "${PROMPT_STYLE}" --n-samples "$2" --temperature 0.6 --top-p 0.9 --max-tokens "${MAX_TOKENS}" \
      --max-model-len "$((MAX_TOKENS + 2048))" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --results-dir "${RESULTS_DIR}/${name}"
  done
}

[[ "${ZERO_SHOT}" == 0 ]] || evaluate zero-shot "${STUDENT}"
for name in ${EXPERIMENTS}; do
  [[ -d "${OUT_ROOT}/${name}" ]] || { echo "no checkpoint ${OUT_ROOT}/${name}: run scripts/span_kd_train.sh first" >&2; exit 2; }
  evaluate "${name}" "${OUT_ROOT}/${name}"
done
