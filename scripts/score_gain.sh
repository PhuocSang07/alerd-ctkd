#!/bin/bash
# Data preparation and answer-gain scoring (needs a GPU with vLLM; not meant for a laptop).
#
# Usage: [STUDENTS=...] [TEACHER=...] [SPAN_MODE=word] [LIMIT=1000] bash scripts/score_gain.sh [WORK_DIR]
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
set -euo pipefail

WORK_DIR=${1:-work}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}
STUDENTS=${STUDENTS:-"openai-community/gpt2-xl meta-llama/Llama-3.2-1B-Instruct"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_LEN=${MAX_LEN:-512}
SPAN_MODE=${SPAN_MODE:-}
LIMIT=${LIMIT:-}
PYTHON=${PYTHON:-python}
[[ -z "${SPAN_MODE}" || "${SPAN_MODE}" == word ]] || { echo "SPAN_MODE must be empty or word" >&2; exit 2; }
[[ -z "${LIMIT}" ]] || WORK_DIR="${WORK_DIR}-limit${LIMIT}"

score() {  # score <model> <side: student|teacher> <data jsonl> <out dir>; writes gain-tokens-<model>.json
  local model=$1 side=$2 data=$3 out=$4 tag started
  tag=$(basename "${model}")
  mkdir -p "${out}"
  started=${SECONDS}
  "${PYTHON}" token_gain.py records --data "${data}" --tokenizer "${model}" --side "${side}" \
    --output "${out}/tokens-${tag}.jsonl"
  "${PYTHON}" answer_gain.py --data-path "${out}/tokens-${tag}.jsonl" --model "${model}" \
    --output "${out}/gain-raw-${tag}.json" --max-model-len "${MAX_MODEL_LEN}"
  "${PYTHON}" token_gain.py pack --records "${out}/tokens-${tag}.jsonl" --gains "${out}/gain-raw-${tag}.json" \
    --model "${model}" --tokenizer "${model}" --output "${out}/gain-tokens-${tag}.json"
  echo "scored ${model} (${side}) in $((SECONDS - started)) s including tokenization"
}

mkdir -p "${WORK_DIR}"
"${PYTHON}" prepare_sra_data.py clean --output "${WORK_DIR}/gsm8k-clean.jsonl"
if [[ -n "${LIMIT}" ]]; then
  head -n "${LIMIT}" "${WORK_DIR}/gsm8k-clean.jsonl" > "${WORK_DIR}/clean.tmp"
  mv "${WORK_DIR}/clean.tmp" "${WORK_DIR}/gsm8k-clean.jsonl"
fi

first=""
for student in ${STUDENTS}; do
  tag=$(basename "${student}")
  out="${WORK_DIR}/${tag}"
  mkdir -p "${out}"
  "${PYTHON}" prepare_sra_data.py build --clean "${WORK_DIR}/gsm8k-clean.jsonl" --student-tokenizer "${student}" \
    --teacher-tokenizer "${TEACHER}" --output "${out}/train-sra.jsonl"
  first=${first:-${out}/train-sra.jsonl}
  score "${student}" student "${out}/train-sra.jsonl" "${out}"
  if [[ -z "${LIMIT}" ]]; then
    "${PYTHON}" score_predictability.py --data "${out}/tokens-${tag}.jsonl" --model "${student}" \
      --output "${out}/predictability.json"
  fi
done

# the teacher's text (its prompt + the shared output) does not depend on the student: score it once
score "${TEACHER}" teacher "${first}" "${WORK_DIR}/$(basename "${TEACHER}")"

if [[ "${SPAN_MODE}" == word ]]; then
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
