#!/bin/bash
# Train and evaluate a set of methods for one student. Needs the same environment as alerd_ctkd_train.sh
# (TEACHER, TEACHER_LAYER, TEACHER_DIM) and alerd_ctkd_eval.sh (SGL_SRC).
#
# Usage: bash scripts/alerd_ctkd_run_all.sh <student> [main|ablations|<method> ...]
#   main        sra, sra-uniform, alerd-ctkd (Table 2 rows), preceded by the zero-shot and teacher evaluations
#   ablations   Tables 3-5 of ALERD, intended for r1-qwen-1.5b
set -euo pipefail

STUDENT=${1:?student}
shift || true
SELECTION=${*:-main}

case "${STUDENT}" in
  r1-qwen-1.5b) MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ;;
  qwen25-7b)    MODEL=Qwen/Qwen2.5-7B-Instruct ;;
  qwen3-8b)     MODEL=Qwen/Qwen3-8B ;;
  *) echo "unknown student ${STUDENT}" >&2; exit 2 ;;
esac

MAIN="sra sra-uniform alerd-ctkd"
ABLATIONS="alerd-ctkd-random alerd-ctkd-pred-easy alerd-ctkd-pred-hard alerd-ctkd-answer-only-unit \
alerd-ctkd-corpus alerd-ctkd-nonorm alerd-ctkd-t2-l0 alerd-ctkd-t2-l05 alerd-ctkd-t2-l1 alerd-ctkd-t4-l05"

METHODS=""
for item in ${SELECTION}; do
  case "${item}" in
    main) METHODS+=" ${MAIN}" ;;
    ablations) METHODS+=" ${ABLATIONS}" ;;
    *) METHODS+=" ${item}" ;;
  esac
done

if [[ " ${SELECTION} " == *" main "* ]]; then
  bash scripts/alerd_ctkd_eval.sh "${MODEL}" "${STUDENT}-zeroshot" "${MODEL}"
  bash scripts/alerd_ctkd_eval.sh "${TEACHER}" "teacher" "${TEACHER}"
fi

for method in ${METHODS}; do
  bash scripts/alerd_ctkd_train.sh "${STUDENT}" "${method}"
  bash scripts/alerd_ctkd_eval.sh "${OUT_ROOT:-outputs}/${STUDENT}/${method}" "${STUDENT}-${method}" "${MODEL}"
done
