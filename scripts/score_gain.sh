#!/bin/bash
# Data preparation and answer-gain scoring (needs a GPU with vLLM; not meant for a laptop).
#
# Usage: [STUDENTS=...] [TEACHER=...] [SPAN_MODE=word] [LIMIT=1000] [DATASET=s1k] [SHARD=0/6] bash scripts/score_gain.sh [WORK_DIR]
#
# For every student: GSM8K train rows in that student's prompt format (train-sra.jsonl), token-level answer gain of the
# frozen student (gain-tokens-<student>.json) and the per-line predictability scores. Once: the token-level answer gain
# of the teacher, scored on the teacher's own prompt (shared by all students). Gain is measured at every token boundary
# with answer_gain.py, one "step" per token (token_gain.py), so a span's or a step's gain is a sum of token
# gains. Authentication for gated models comes from the HF_TOKEN environment variable, never from this script.
#
# SPAN_MODE=word (one tokenizer for student and teacher, e.g. STUDENTS=Qwen/Qwen2.5-1.5B-Instruct
# TEACHER=Qwen/Qwen2.5-14B-Instruct) also writes, per student, the word-span index spans-<teacher>.jsonl and
# span-gains.json: the student's span gains (for the CE weights) and teacher minus student (for the KL weights).
# LIMIT=N keeps the first N problems, writes to WORK_DIR-limitN and skips predictability: a cost measurement. The number
# of probes is printed by token_gain.py / span_index.py, the wall time of each scoring run by answer_gain.py and below.
# DATASET=s1k scores s1K-1.1 (the deepseek_attempt write-ups, MAX_LEN 1024 tokens) instead of GSM8K. SHARD=i/n scores every
# n-th row (starting at i) into WORK_DIR-shard<i>of<n> and stops after the two gain-tokens files: scripts/modal_score_gain.py
# runs the shards in parallel and merges them (token_gain.py merge), then builds the span index and the signals.
set -euo pipefail

WORK_DIR=${1:-work}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}
STUDENTS=${STUDENTS:-"openai-community/gpt2-xl meta-llama/Llama-3.2-1B-Instruct"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
DATASET=${DATASET:-gsm8k}
SHARD=${SHARD:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}  # raise for a teacher whose weights fill a third of the GPU
SPAN_MODE=${SPAN_MODE:-}
LIMIT=${LIMIT:-}
TEACHER_MAX_LEN=${TEACHER_MAX_LEN:-2560}
PYTHON=${PYTHON:-python}
[[ -z "${SPAN_MODE}" || "${SPAN_MODE}" == word ]] || { echo "SPAN_MODE must be empty or word" >&2; exit 2; }
[[ "${DATASET}" == gsm8k || "${DATASET}" == s1k ]] || { echo "DATASET must be gsm8k or s1k" >&2; exit 2; }
if [[ "${DATASET}" == s1k ]]; then MAX_LEN=${MAX_LEN:-1024}; else MAX_LEN=${MAX_LEN:-512}; fi
[[ -z "${SHARD}" || "${SHARD}" =~ ^[0-9]+/[1-9][0-9]*$ ]] || { echo "SHARD must look like 0/6" >&2; exit 2; }
[[ -z "${SHARD}" || "${SHARD%/*}" -lt "${SHARD#*/}" ]] || { echo "SHARD index must be below the shard count" >&2; exit 2; }
[[ -z "${LIMIT}" ]] || WORK_DIR="${WORK_DIR}-limit${LIMIT}"
[[ -z "${SHARD}" ]] || WORK_DIR="${WORK_DIR}-shard${SHARD%/*}of${SHARD#*/}"
CLEAN="${WORK_DIR}/${DATASET}-clean.jsonl"

score() {  # score <model> <side: student|teacher> <data jsonl> <out dir>; writes gain-tokens-<model>.json
  local model=$1 side=$2 data=$3 out=$4 tag started
  tag=$(basename "${model}")
  mkdir -p "${out}"
  started=${SECONDS}
  "${PYTHON}" token_gain.py records --data "${data}" --tokenizer "${model}" --side "${side}" \
    --output "${out}/tokens-${tag}.jsonl"
  "${PYTHON}" answer_gain.py --data-path "${out}/tokens-${tag}.jsonl" --model "${model}" \
    --output "${out}/gain-raw-${tag}.json" --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  "${PYTHON}" token_gain.py pack --records "${out}/tokens-${tag}.jsonl" --gains "${out}/gain-raw-${tag}.json" \
    --model "${model}" --tokenizer "${model}" --output "${out}/gain-tokens-${tag}.json"
  echo "scored ${model} (${side}) in $((SECONDS - started)) s including tokenization"
}

mkdir -p "${WORK_DIR}"
if [[ "${DATASET}" == s1k ]]; then
  "${PYTHON}" prepare_sra_data.py clean-s1k --output "${CLEAN}"
else
  "${PYTHON}" prepare_sra_data.py clean --output "${CLEAN}"
fi
if [[ -n "${LIMIT}" ]]; then
  head -n "${LIMIT}" "${CLEAN}" > "${WORK_DIR}/clean.tmp"
  mv "${WORK_DIR}/clean.tmp" "${CLEAN}"
fi

first=""
for student in ${STUDENTS}; do
  tag=$(basename "${student}")
  out="${WORK_DIR}/${tag}"
  mkdir -p "${out}"
  build_limits=()
  [[ "${DATASET}" == s1k ]] && build_limits=(--max-len "${MAX_LEN}" --teacher-max-len "${TEACHER_MAX_LEN}")
  "${PYTHON}" prepare_sra_data.py build --clean "${CLEAN}" --student-tokenizer "${student}" \
    --teacher-tokenizer "${TEACHER}" "${build_limits[@]}" --output "${out}/train-sra.jsonl"
  if [[ -n "${SHARD}" ]]; then  # every n-th row of the (filtered) training file
    awk -v n="${SHARD#*/}" -v i="${SHARD%/*}" '(NR - 1) % n == i' "${out}/train-sra.jsonl" > "${out}/shard.tmp"
    mv "${out}/shard.tmp" "${out}/train-sra.jsonl"
  fi
  first=${first:-${out}/train-sra.jsonl}
  score "${student}" student "${out}/train-sra.jsonl" "${out}"
  if [[ -z "${LIMIT}" && -z "${SHARD}" ]]; then
    "${PYTHON}" score_predictability.py --data "${out}/tokens-${tag}.jsonl" --model "${student}" \
      --output "${out}/predictability.json"
  fi
done

# the teacher's text (its prompt + the shared output) does not depend on the student: score it once
score "${TEACHER}" teacher "${first}" "${WORK_DIR}/$(basename "${TEACHER}")"

if [[ "${SPAN_MODE}" == word && -z "${SHARD}" ]]; then
  teacher_tag=$(basename "${TEACHER}")
  for student in ${STUDENTS}; do
    tag=$(basename "${student}")
    out="${WORK_DIR}/${tag}"
    "${PYTHON}" span_index.py --data "${out}/train-sra.jsonl" --student-tokenizer "${student}" \
      --teacher-tokenizer "${TEACHER}" --span-mode word --max-len "${MAX_LEN}" --output "${out}/spans-${teacher_tag}.jsonl"
    "${PYTHON}" token_gain.py signals --spans "${out}/spans-${teacher_tag}.jsonl" \
      --student-gains "${out}/gain-tokens-${tag}.json" \
      --teacher-gains "${WORK_DIR}/${teacher_tag}/gain-tokens-${teacher_tag}.json" --output "${out}/span-gains.json"
  done
fi
