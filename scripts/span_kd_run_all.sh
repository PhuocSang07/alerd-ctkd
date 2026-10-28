#!/bin/bash
# Whole pipeline on one machine: answer-gain scoring -> five training runs -> evaluation.
#
# Usage: [DATASET=gsm8k|s1k|tsdkd] [LIMIT=1000] [EXPERIMENTS="..."] [SKIP_SCORING=1] bash scripts/span_kd_run_all.sh
#
# 1. scripts/score_gain.sh with SPAN_MODE=word (word spans, student and teacher gain, span-gains.json). The cost is the
#    teacher's answer-gain probes (one per response token): measured ~38 probes/s for Qwen2.5-14B on one A100-80GB, i.e.
#    ~42 min for 1,000 GSM8K problems and over 3 hours for the 830 s1K problems; SKIP_SCORING=1 reuses an earlier run.
# 2. scripts/span_kd_train.sh on that output.
# 3. scripts/span_kd_eval.sh (GSM8K, MATH500, AIME24, AIME25 at 4096 new tokens; ZERO_SHOT=1 adds the student).
# DATASET=tsdkd defaults: the first 2000 rows (LIMIT=), 2 epochs, effective batch 64 in micro batches of 8, the five
# experiments plus kd-token-mean, and the zero-shot student in the evaluation.
# Every setting of those scripts (STUDENT, TEACHER, EPOCHS, LEARNING_RATE, ...) can be passed through the environment.
set -euo pipefail

export STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
export TEACHER=${TEACHER:-Qwen/Qwen2.5-14B-Instruct}
DATASET=${DATASET:-gsm8k}
if [[ "${DATASET}" == tsdkd ]]; then LIMIT=${LIMIT-2000}; else LIMIT=${LIMIT:-}; fi  # tsdkd: first 2000 rows; LIMIT= for all
SKIP_SCORING=${SKIP_SCORING:-0}
student_tag=$(basename "${STUDENT}")

work="${WORK_ROOT:-work}"
[[ "${DATASET}" == gsm8k ]] || work="${work}-${DATASET}"
[[ -z "${LIMIT}" ]] || work="${work}-limit${LIMIT}"

if [[ "${SKIP_SCORING}" == 0 ]]; then
  DATASET="${DATASET}" LIMIT="${LIMIT}" SPAN_MODE=word GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}" \
    bash scripts/score_gain.sh "${work%-limit*}"
fi

export DATA_DIR="${DATA_DIR:-${work}/${student_tag}}"
export OUT_ROOT="${OUT_ROOT:-outputs/span-kd-${DATASET}}"
export RESULTS_DIR="${RESULTS_DIR:-results/span-kd-${DATASET}}"
[[ "${DATASET}" != s1k ]] || export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"  # ~1,000-token samples
if [[ "${DATASET}" == tsdkd ]]; then  # the TSD-KD schedule: effective batch 64 (8 micro batches of 8 on an H200 141GB)
  export EPOCHS="${EPOCHS:-2}" BATCH_SIZE="${BATCH_SIZE:-64}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
  export EXPERIMENTS="${EXPERIMENTS:-sft kd-token kd-token-mean gain-ce gain-kd gain-both}"
  export ZERO_SHOT="${ZERO_SHOT:-1}"
fi
bash scripts/span_kd_train.sh
bash scripts/span_kd_eval.sh
