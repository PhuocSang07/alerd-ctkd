#!/bin/bash
# Data preparation and answer-gain scoring (needs a GPU with vLLM; not meant for a laptop).
#
# Usage: bash scripts/score_gain.sh [WORK_DIR]
#
# For every student: GSM8K train rows in that student's prompt format (train-sra.jsonl), token-level answer gain of the
# frozen student (gain-tokens-<student>.json) and the per-line predictability scores. Once: the token-level answer gain
# of the teacher, scored on the teacher's own prompt (shared by all students). Gain is measured at every token boundary
# with answer_gain.py, one "step" per token (token_gain.py), so a span's or a step's gain is a sum of token
# gains. Authentication for gated models comes from the HF_TOKEN environment variable, never from this script.
set -euo pipefail

WORK_DIR=${1:-work}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}
STUDENTS=${STUDENTS:-"openai-community/gpt2-xl meta-llama/Llama-3.2-1B-Instruct"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
PYTHON=${PYTHON:-python}

score() {  # score <model> <side: student|teacher> <data jsonl> <out dir>; writes gain-tokens-<model>.json
  local model=$1 side=$2 data=$3 out=$4 tag
  tag=$(basename "${model}")
  mkdir -p "${out}"
  "${PYTHON}" token_gain.py records --data "${data}" --tokenizer "${model}" --side "${side}" \
    --output "${out}/tokens-${tag}.jsonl"
  "${PYTHON}" answer_gain.py --data-path "${out}/tokens-${tag}.jsonl" --model "${model}" \
    --output "${out}/gain-raw-${tag}.json" --max-model-len "${MAX_MODEL_LEN}"
  "${PYTHON}" token_gain.py pack --records "${out}/tokens-${tag}.jsonl" --gains "${out}/gain-raw-${tag}.json" \
    --model "${model}" --tokenizer "${model}" --output "${out}/gain-tokens-${tag}.json"
}

mkdir -p "${WORK_DIR}"
"${PYTHON}" prepare_sra_data.py clean --output "${WORK_DIR}/gsm8k-clean.jsonl"

first=""
for student in ${STUDENTS}; do
  tag=$(basename "${student}")
  out="${WORK_DIR}/${tag}"
  mkdir -p "${out}"
  "${PYTHON}" prepare_sra_data.py build --clean "${WORK_DIR}/gsm8k-clean.jsonl" --student-tokenizer "${student}" \
    --teacher-tokenizer "${TEACHER}" --output "${out}/train-sra.jsonl"
  first=${first:-${out}/train-sra.jsonl}
  score "${student}" student "${out}/train-sra.jsonl" "${out}"
  "${PYTHON}" score_predictability.py --data "${out}/tokens-${tag}.jsonl" --model "${student}" \
    --output "${out}/predictability.json"
done

# the teacher's text (its prompt + the shared output) does not depend on the student: score it once
score "${TEACHER}" teacher "${first}" "${WORK_DIR}/$(basename "${TEACHER}")"
