#!/bin/bash
# Train one student with one method (3 epochs, lr 5e-5 -> 1e-5 cosine, warm-up 10%, effective batch 32, full fine-tuning).
#
# Usage:
#   bash scripts/alerd_ctkd_train.sh <student> <method> [seed]
#
#   student: gpt2-xl | llama-3.2-1b
#   method:  sft | sra | sra-uniform | alerd-ctkd                      main table
#            alerd-ctkd-gain-student | alerd-ctkd-gain-teacher | alerd-ctkd-gain-diff     gain source (span unit)
#            alerd-ctkd-unit-step                                      gain unit (spans inherit the gain of their step)
#            alerd-ctkd-random | alerd-ctkd-pred-easy | alerd-ctkd-pred-hard | alerd-ctkd-answer-only-unit
#            alerd-ctkd-corpus | alerd-ctkd-nonorm                     step score, normalization
#            alerd-ctkd-t2-l0 | alerd-ctkd-t2-l05 | alerd-ctkd-t2-l1 | alerd-ctkd-t4-l05   shrinkage, temperature
#
# Environment (defaults in brackets):
#   TEACHER [Qwen/Qwen2.5-7B-Instruct], TEACHER_LAYER [28], TEACHER_DIM [3584]   distilled teacher layer, 1-indexed
#   GAIN_SOURCE [student] student | teacher | diff, GAIN_UNIT [span] span | step    for every alerd-ctkd method
#   WORK_DIR [work], OUT_ROOT [outputs], MAX_LEN [512], BATCH_SIZE [8], GRAD_ACCUM [4], SEED via the third argument
#   STUDENT_DEVICE [cuda:0], TEACH_DEVICE [cuda:1], DUMP_SPAN_WEIGHTS (set to write per-span attention and omega)
#
# Needs work/<student>/{train-sra.jsonl,gain-tokens-<student>.json[,predictability.json]} and
# work/<teacher>/gain-tokens-<teacher>.json from scripts/score_gain.sh. The span index is built on first use.
# The Hugging Face token is read from HF_TOKEN.
set -euo pipefail

STUDENT=${1:?student}
METHOD=${2:?method}
SEED=${3:-42}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}
TEACHER_LAYER=${TEACHER_LAYER:-28}
TEACHER_DIM=${TEACHER_DIM:-3584}
TEACHER_TOKENIZER=${TEACHER_TOKENIZER:-$TEACHER}
TEACHER_TYPE=${TEACHER_TYPE:-qwen}
GAIN_SOURCE=${GAIN_SOURCE:-student}
GAIN_UNIT=${GAIN_UNIT:-span}
WORK_DIR=${WORK_DIR:-work}
OUT_ROOT=${OUT_ROOT:-outputs}
MAX_LEN=${MAX_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}
TEACH_DEVICE=${TEACH_DEVICE:-cuda:1}
PYTHON=${PYTHON:-python}

case "${STUDENT}" in
  gpt2-xl)      MODEL=openai-community/gpt2-xl;          LAYER=48; STUDENT_TYPE=gpt2 ;;
  llama-3.2-1b) MODEL=meta-llama/Llama-3.2-1B-Instruct;  LAYER=16; STUDENT_TYPE=llama ;;
  *) echo "unknown student ${STUDENT}" >&2; exit 2 ;;
esac

DATA="${WORK_DIR}/$(basename "${MODEL}")"
SPANS="${DATA}/spans-$(basename "${TEACHER_TOKENIZER}").jsonl"
STUDENT_GAIN="${DATA}/gain-tokens-$(basename "${MODEL}").json"
TEACHER_GAIN="${WORK_DIR}/$(basename "${TEACHER}")/gain-tokens-$(basename "${TEACHER}").json"
OUT="${OUT_ROOT}/${STUDENT}/${METHOD}/seed-${SEED}"

# alerd-ctkd family: gain source and unit from the environment unless the method fixes them (the gain-source ablations
# use the span unit, the unit ablation keeps the source)
SOURCE=${GAIN_SOURCE}
UNIT=${GAIN_UNIT}
case "${METHOD}" in
  alerd-ctkd-gain-student) SOURCE=student; UNIT=span ;;
  alerd-ctkd-gain-teacher) SOURCE=teacher; UNIT=span ;;
  alerd-ctkd-gain-diff)    SOURCE=diff; UNIT=span ;;
  alerd-ctkd-unit-step)    UNIT=step ;;
esac
gain_args() {
  echo "--spans_file ${SPANS} --gain_source ${SOURCE} --gain_unit ${UNIT} --student_gain_file ${STUDENT_GAIN}" \
       "--teacher_gain_file ${TEACHER_GAIN} --alloc_tau 1.0 --alloc_lambda 0.5 --alloc_norm per-trace"
}
require_gain_files() {
  local missing=0
  if [[ "${SOURCE}" == student || "${SOURCE}" == diff ]] && [[ ! -f "${STUDENT_GAIN}" ]]; then
    echo "missing ${STUDENT_GAIN} (gain source ${SOURCE}); run scripts/score_gain.sh" >&2; missing=1
  fi
  if [[ "${SOURCE}" == teacher || "${SOURCE}" == diff ]] && [[ ! -f "${TEACHER_GAIN}" ]]; then
    echo "missing ${TEACHER_GAIN} (gain source ${SOURCE}); run scripts/score_gain.sh" >&2; missing=1
  fi
  [[ ${missing} -eq 0 ]] || exit 3
}

KD=1
case "${METHOD}" in
  sft)                         KD=0; WEIGHTS="" ;;
  sra)                         WEIGHTS="--span_weight_source attention" ;;
  sra-uniform)                 WEIGHTS="--span_weight_source uniform --spans_file ${SPANS}" ;;
  alerd-ctkd|alerd-ctkd-gain-student|alerd-ctkd-gain-teacher|alerd-ctkd-gain-diff|alerd-ctkd-unit-step)
                               require_gain_files; WEIGHTS="--span_weight_source alerd $(gain_args)" ;;
  alerd-ctkd-random)           require_gain_files; WEIGHTS="--span_weight_source random $(gain_args)" ;;
  alerd-ctkd-pred-easy)        WEIGHTS="--span_weight_source predictability-easy $(gain_args) --predictability_file ${DATA}/predictability.json" ;;
  alerd-ctkd-pred-hard)        WEIGHTS="--span_weight_source predictability-hard $(gain_args) --predictability_file ${DATA}/predictability.json" ;;
  alerd-ctkd-answer-only-unit) require_gain_files; WEIGHTS="--span_weight_source alerd $(gain_args) --answer_only_unit True" ;;
  alerd-ctkd-corpus)           require_gain_files; WEIGHTS="--span_weight_source alerd $(gain_args) --alloc_norm corpus" ;;
  alerd-ctkd-nonorm)           require_gain_files; WEIGHTS="--span_weight_source alerd $(gain_args) --alloc_norm none" ;;
  alerd-ctkd-t[0-9]-l0|alerd-ctkd-t[0-9]-l05|alerd-ctkd-t[0-9]-l1)
    TAU=${METHOD#alerd-ctkd-t}; TAU=${TAU%%-*}
    LAMBDA_CODE=${METHOD##*-l}
    case "${LAMBDA_CODE}" in 0) LAMBDA=0.0 ;; 05) LAMBDA=0.5 ;; 1) LAMBDA=1.0 ;; esac
    require_gain_files
    WEIGHTS="--span_weight_source alerd $(gain_args) --alloc_tau ${TAU} --alloc_lambda ${LAMBDA}" ;;
  *) echo "unknown method ${METHOD}" >&2; exit 2 ;;
esac

# predictability scores are per solution line, so they only need the span index and the student's score file
if [[ "${METHOD}" == alerd-ctkd-pred-* && ! -f "${DATA}/predictability.json" ]]; then
  echo "missing ${DATA}/predictability.json; run scripts/score_gain.sh" >&2; exit 3
fi

if [[ "${WEIGHTS}" == *"--spans_file"* && ! -f "${SPANS}" ]]; then
  "${PYTHON}" span_index.py --data "${DATA}/train-sra.jsonl" --student-tokenizer "${MODEL}" \
    --teacher-tokenizer "${TEACHER_TOKENIZER}" --max-len "${MAX_LEN}" --output "${SPANS}"
fi

mkdir -p "${OUT}"
OPTS=""
OPTS+=" --train_data ${DATA}/train-sra.jsonl"
OPTS+=" --max_len ${MAX_LEN} --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM}"
OPTS+=" --num_train_epochs 3 --learning_rate 5e-5 --min_learning_rate 1e-5 --warmup_ratio 0.1"
OPTS+=" --student_model ${MODEL} --student_tokenizer ${MODEL}"
OPTS+=" --student_device ${STUDENT_DEVICE} --student_dtype float32 --use_lora False"
OPTS+=" --output_dir ${OUT} --seed ${SEED} --student_model_type ${STUDENT_TYPE}"
if [[ ${KD} -eq 1 ]]; then
  OPTS+=" --hard_label_loss_weight 0.5 --span_loss True --span_weight_pooling True --span_loss_weight True --p 1.0"
  OPTS+=" --orthogonal True"
  OPTS+=" --teacher_layers_mapping ${TEACHER_LAYER} --student_encoder_layers_finetuned ${LAYER}"
  OPTS+=" --n_encoder_finetuned ${LAYER} --hidden_loss_weights 1 --teacher_embedding_dimension ${TEACHER_DIM}"
  OPTS+=" --teacher_model ${TEACHER} --teacher_tokenizer ${TEACHER_TOKENIZER} --teach_device ${TEACH_DEVICE}"
  OPTS+=" --teacher_model_type ${TEACHER_TYPE} ${WEIGHTS}"
  if [[ -n "${DUMP_SPAN_WEIGHTS:-}" ]]; then
    OPTS+=" --dump_span_weights ${OUT}/span_weights.jsonl"
  fi
else
  OPTS+=" --hard_label_loss_weight 1.0"  # no distillation term: the teacher is not loaded
fi

"${PYTHON}" run_distill_llm.py ${OPTS} >> "${OUT}/train.log" 2>&1
echo "trained ${STUDENT}/${METHOD}/seed-${SEED} -> ${OUT}"
