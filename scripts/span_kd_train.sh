#!/bin/bash
# Span-weighted CE + KD distillation, one run per experiment (needs a GPU that holds the teacher and the student).
#
# Usage: [DATA_DIR=...] [STUDENT=...] [TEACHER=...] [OUT_ROOT=...] [EXPERIMENTS="sft kd-token ..."] bash scripts/span_kd_train.sh
#
# DATA_DIR is the output of scripts/score_gain.sh (SPAN_MODE=word) for the student, i.e. it holds train-sra.jsonl,
# spans-<teacher>.jsonl and span-gains.json. Every experiment writes a merged bf16 checkpoint to OUT_ROOT/<name>/ plus
# train_stats.jsonl (loss, CE, KD, grad norm per optimizer step) and spans.jsonl (CE, KD and masses of every span).
#
# name            CE masses  KD masses  beta  lambda
#   sft           token      -          0     0.5    CE only, the teacher is not loaded
#   kd-token      token      token      0.5   0.5    plain token-mean CE + KD
#   kd-token-mean token      token      0.5   0.5    kd-token averaged over the tokens of the batch (the baselines'
#                                                    token mean); the other rows average over the samples
#   kd-uniform    uniform    uniform    0.5   0.5    every span the same mass
#   gain-ce       student    token      0.5   0.5    student gain on the CE only
#   gain-kd       token      diff       0.5   0.5    teacher - student gain on the KD only
#   gain-both     student    diff       0.5   0.5    the proposed pairing
#   gain-student  student    student    0.5   0.5
#   gain-diff     diff       diff       0.5   0.5
#   gain-both-lam1 student   diff       0.5   1.0    full gain weights, no uniform share
# (the same table is EXPERIMENTS in scripts/modal_train_kd.py)
set -euo pipefail

STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
TEACHER=${TEACHER:-Qwen/Qwen2.5-14B-Instruct}
DATA_DIR=${DATA_DIR:-work/$(basename "${STUDENT}")}
OUT_ROOT=${OUT_ROOT:-outputs/span-kd}
EXPERIMENTS=${EXPERIMENTS:-"sft kd-token gain-ce gain-kd gain-both"}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}             # effective batch (micro batches are accumulated)
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}  # 4 for the ~1,000-token s1K samples, 8 is fine for GSM8K
LEARNING_RATE=${LEARNING_RATE:-1e-4}
KD_TOP_K=${KD_TOP_K:-50}
SEED=${SEED:-10}
PYTHON=${PYTHON:-python}

teacher_tag=$(basename "${TEACHER}")
for file in train-sra.jsonl "spans-${teacher_tag}.jsonl" span-gains.json; do
  [[ -f "${DATA_DIR}/${file}" ]] || { echo "missing ${DATA_DIR}/${file}: run scripts/score_gain.sh first" >&2; exit 2; }
done

for name in ${EXPERIMENTS}; do
  mean=sample
  case "${name}" in
    sft)            ce=token;   kd=token;   beta=0;   lam=0.5 ;;
    kd-token)       ce=token;   kd=token;   beta=0.5; lam=0.5 ;;
    kd-token-mean)  ce=token;   kd=token;   beta=0.5; lam=0.5; mean=token ;;
    kd-uniform)     ce=uniform; kd=uniform; beta=0.5; lam=0.5 ;;
    gain-ce)        ce=student; kd=token;   beta=0.5; lam=0.5 ;;
    gain-kd)        ce=token;   kd=diff;    beta=0.5; lam=0.5 ;;
    gain-both)      ce=student; kd=diff;    beta=0.5; lam=0.5 ;;
    gain-student)   ce=student; kd=student; beta=0.5; lam=0.5 ;;
    gain-diff)      ce=diff;    kd=diff;    beta=0.5; lam=0.5 ;;
    gain-both-lam1) ce=student; kd=diff;    beta=0.5; lam=1.0 ;;
    *) echo "unknown experiment ${name}" >&2; exit 2 ;;
  esac
  out="${OUT_ROOT}/${name}"
  mkdir -p "${out}"
  teacher_args=()
  [[ "${beta}" != 0 ]] && teacher_args=(--teacher-model "${TEACHER}")
  started=${SECONDS}
  "${PYTHON}" span_kd_train.py --train-data "${DATA_DIR}/train-sra.jsonl" \
    --spans-file "${DATA_DIR}/spans-${teacher_tag}.jsonl" --gains-file "${DATA_DIR}/span-gains.json" \
    --student-model "${STUDENT}" "${teacher_args[@]}" --output-dir "${out}" \
    --ce-weights "${ce}" --kd-weights "${kd}" --kd-beta "${beta}" --alloc-lambda "${lam}" --kd-top-k "${KD_TOP_K}" \
    --loss-mean "${mean}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" --seed "${SEED}" --dump-spans "${out}/spans.jsonl"
  echo "trained ${name} in $((SECONDS - started)) s -> ${out}"
done
