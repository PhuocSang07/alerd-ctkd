#!/bin/bash
# CPU-only look at the training data: the Math CoT rows of the TSD-KD release, in the student's and teacher's prompts.
#
# Usage: [TSDKD_SOURCE=<Hub id or local jsonl>] [NUM_SAMPLES=K] [STUDENT=...] [TEACHER=...] [OUT_DIR=work/tsdkd] bash scripts/prepare_tsdkd.sh
#
# Downloads the filtered math rows (phuocsang/UltraInteract-math-Qwen2.5-14B-Instruct-Gen, 22,543 rows; a raw release is
# filtered on the fly) into tsdkd-clean.jsonl, then writes train-sra-all.jsonl with both prompts, drops the rows longer than
# MAX_LEN tokens and keeps NUM_SAMPLES random rows of the rest (default: all). The printed counts and token lengths give the
# size of the answer-gain scoring (one probe per response token, for each of the two models) before any GPU is rented.
# scripts/score_gain.sh DATASET=tsdkd repeats these two steps itself, so this script is only needed for the numbers.
set -euo pipefail

STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
TEACHER=${TEACHER:-Qwen/Qwen2.5-14B-Instruct}
OUT_DIR=${OUT_DIR:-work/tsdkd}
MAX_LEN=${MAX_LEN:-1024}
NUM_SAMPLES=${NUM_SAMPLES:-0}  # 0 = every row that fits MAX_LEN
SAMPLE_SEED=${SAMPLE_SEED:-10}
PYTHON=${PYTHON:-python}

mkdir -p "${OUT_DIR}"
"${PYTHON}" prepare_sra_data.py clean-tsdkd ${TSDKD_SOURCE:+--dataset "${TSDKD_SOURCE}"} --output "${OUT_DIR}/tsdkd-clean.jsonl"
"${PYTHON}" prepare_sra_data.py build --clean "${OUT_DIR}/tsdkd-clean.jsonl" --student-tokenizer "${STUDENT}" \
  --teacher-tokenizer "${TEACHER}" --max-len "${MAX_LEN}" --teacher-max-len "${MAX_LEN}" \
  --num-samples "${NUM_SAMPLES}" --sample-seed "${SAMPLE_SEED}" --output "${OUT_DIR}/train-sra-all.jsonl"
