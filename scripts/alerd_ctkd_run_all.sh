#!/bin/bash
# Train and evaluate a set of methods for one student (alerd_ctkd_train.sh and alerd_ctkd_eval.sh; the environment of
# both applies: TEACHER*, GAIN_SOURCE and GAIN_UNIT for training).
#
# Usage: bash scripts/alerd_ctkd_run_all.sh <student> [main|ablations|<method> ...]
#   main        sft, sra, sra-uniform, alerd-ctkd on SEEDS [42 43 44], after the zero-shot and teacher evaluations
#   ablations   the gain-source, gain-unit and Table 3-5 variants on ABLATION_SEED [42], intended for llama-3.2-1b
# Results are tagged <student>-<method>-s<seed>, <student>-zeroshot and teacher under ${RESULTS_DIR:-results}.
set -euo pipefail

STUDENT=${1:?student}
shift || true
SELECTION=${*:-main}
SEEDS=${SEEDS:-"42 43 44"}
ABLATION_SEED=${ABLATION_SEED:-42}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}

case "${STUDENT}" in
  gpt2-xl)      MODEL=openai-community/gpt2-xl ;;
  llama-3.2-1b) MODEL=meta-llama/Llama-3.2-1B-Instruct ;;
  *) echo "unknown student ${STUDENT}" >&2; exit 2 ;;
esac

MAIN="sft sra sra-uniform alerd-ctkd"
ABLATIONS="alerd-ctkd-gain-student alerd-ctkd-gain-teacher alerd-ctkd-gain-diff alerd-ctkd-unit-step \
alerd-ctkd-random alerd-ctkd-pred-easy alerd-ctkd-pred-hard alerd-ctkd-answer-only-unit \
alerd-ctkd-corpus alerd-ctkd-nonorm alerd-ctkd-t2-l0 alerd-ctkd-t2-l05 alerd-ctkd-t2-l1 alerd-ctkd-t4-l05"

if [[ " ${SELECTION} " == *" main "* ]]; then
  bash scripts/alerd_ctkd_eval.sh "${MODEL}" "${STUDENT}-zeroshot" "${MODEL}"
  bash scripts/alerd_ctkd_eval.sh "${TEACHER}" "teacher" "${TEACHER}"
fi

run() {  # run <method> <seed>
  bash scripts/alerd_ctkd_train.sh "${STUDENT}" "$1" "$2"
  bash scripts/alerd_ctkd_eval.sh "${OUT_ROOT:-outputs}/${STUDENT}/$1/seed-$2" "${STUDENT}-$1-s$2" "${MODEL}"
}

for item in ${SELECTION}; do
  case "${item}" in
    main)      for method in ${MAIN}; do for seed in ${SEEDS}; do run "${method}" "${seed}"; done; done ;;
    ablations) for method in ${ABLATIONS}; do run "${method}" "${ABLATION_SEED}"; done ;;
    *)         for seed in ${SEEDS}; do run "${item}" "${seed}"; done ;;
  esac
done
