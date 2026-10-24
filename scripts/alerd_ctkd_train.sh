#!/bin/bash
# Train one student with one span-weight method (ALERD recipe: 3 epochs, lr 5e-5 -> 1e-5, batch 1 x 32).
#
# Usage:
#   TEACHER=<hf id or path> TEACHER_LAYER=<1-indexed layer> TEACHER_DIM=<hidden size> \
#   bash scripts/alerd_ctkd_train.sh <student> <method> [seed]
#
#   student: r1-qwen-1.5b (full fine-tuning) | qwen25-7b | qwen3-8b (LoRA)
#   method:  sra | sra-uniform | alerd-ctkd                           main table
#            alerd-ctkd-random | alerd-ctkd-pred-easy | alerd-ctkd-pred-hard | alerd-ctkd-answer-only-unit
#            alerd-ctkd-corpus | alerd-ctkd-nonorm                    ablations (step score, normalization)
#            alerd-ctkd-t2-l0 | alerd-ctkd-t2-l05 | alerd-ctkd-t2-l1 | alerd-ctkd-t4-l05   (shrinkage, temperature)
#
# Needs work/<student model name>/{train-sra.jsonl,gain.json[,predictability.json]} from scripts/score_gain.sh.
# The span index is built on first use. The Hugging Face token is read from HF_TOKEN.
set -euo pipefail

STUDENT=${1:?student}
METHOD=${2:?method}
SEED=${3:-42}
: "${TEACHER:?set TEACHER}" "${TEACHER_LAYER:?set TEACHER_LAYER}" "${TEACHER_DIM:?set TEACHER_DIM}"
TEACHER_TOKENIZER=${TEACHER_TOKENIZER:-$TEACHER}
TEACHER_TYPE=${TEACHER_TYPE:-llama}
WORK_DIR=${WORK_DIR:-work}
OUT_ROOT=${OUT_ROOT:-outputs}
MAX_LEN=${MAX_LEN:-2500}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}
TEACH_DEVICE=${TEACH_DEVICE:-cuda:1}

case "${STUDENT}" in
  r1-qwen-1.5b) MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; LAYER=28; PRECISION="--student_dtype float32 --use_lora False" ;;
  qwen25-7b)    MODEL=Qwen/Qwen2.5-7B-Instruct;                  LAYER=28; PRECISION="--student_dtype bfloat16 --use_lora True" ;;
  qwen3-8b)     MODEL=Qwen/Qwen3-8B;                             LAYER=36; PRECISION="--student_dtype bfloat16 --use_lora True" ;;
  *) echo "unknown student ${STUDENT}" >&2; exit 2 ;;
esac

DATA="${WORK_DIR}/$(basename "${MODEL}")"
SPANS="${DATA}/spans-$(basename "${TEACHER_TOKENIZER}").jsonl"
OUT="${OUT_ROOT}/${STUDENT}/${METHOD}"

# span weights: source plus allocation hyper-parameters (tau=1, lambda=0.5, per-trace normalization by default)
ALLOC="--spans_file ${SPANS} --gain_file ${DATA}/gain.json --alloc_tau 1.0 --alloc_lambda 0.5 --alloc_norm per-trace"
case "${METHOD}" in
  sra)                         WEIGHTS="--span_weight_source attention" ;;
  sra-uniform)                 WEIGHTS="--span_weight_source uniform --spans_file ${SPANS}" ;;
  alerd-ctkd)                  WEIGHTS="--span_weight_source alerd ${ALLOC}" ;;
  alerd-ctkd-random)           WEIGHTS="--span_weight_source random ${ALLOC}" ;;
  alerd-ctkd-pred-easy)        WEIGHTS="--span_weight_source predictability-easy ${ALLOC} --predictability_file ${DATA}/predictability.json" ;;
  alerd-ctkd-pred-hard)        WEIGHTS="--span_weight_source predictability-hard ${ALLOC} --predictability_file ${DATA}/predictability.json" ;;
  alerd-ctkd-answer-only-unit) WEIGHTS="--span_weight_source alerd ${ALLOC} --answer_only_unit True" ;;
  alerd-ctkd-corpus)           WEIGHTS="--span_weight_source alerd ${ALLOC} --alloc_norm corpus" ;;
  alerd-ctkd-nonorm)           WEIGHTS="--span_weight_source alerd ${ALLOC} --alloc_norm none" ;;
  alerd-ctkd-t[0-9]-l0|alerd-ctkd-t[0-9]-l05|alerd-ctkd-t[0-9]-l1)
    TAU=${METHOD#alerd-ctkd-t}; TAU=${TAU%%-*}
    LAMBDA_CODE=${METHOD##*-l}
    case "${LAMBDA_CODE}" in 0) LAMBDA=0.0 ;; 05) LAMBDA=0.5 ;; 1) LAMBDA=1.0 ;; esac
    WEIGHTS="--span_weight_source alerd ${ALLOC} --alloc_tau ${TAU} --alloc_lambda ${LAMBDA}" ;;
  *) echo "unknown method ${METHOD}" >&2; exit 2 ;;
esac

if [[ "${WEIGHTS}" == *"--spans_file"* && ! -f "${SPANS}" ]]; then
  "${PYTHON:-python}" span_index.py --data "${DATA}/train-sra.jsonl" --student-tokenizer "${MODEL}" \
    --teacher-tokenizer "${TEACHER_TOKENIZER}" --max-len "${MAX_LEN}" --output "${SPANS}"
fi

mkdir -p "${OUT}"
OPTS=""
OPTS+=" --train_data ${DATA}/train-sra.jsonl"
OPTS+=" --max_len ${MAX_LEN} --batch_size 1 --grad_accum_steps 32"
OPTS+=" --num_train_epochs 3 --learning_rate 5e-5 --min_learning_rate 1e-5 --warmup_ratio 0.1"
OPTS+=" --hard_label_loss_weight 0.5 --span_loss True --span_weight_pooling True --span_loss_weight True --p 1.0"
OPTS+=" --orthogonal True"
OPTS+=" --teacher_layers_mapping ${TEACHER_LAYER} --student_encoder_layers_finetuned ${LAYER}"
OPTS+=" --n_encoder_finetuned ${LAYER} --hidden_loss_weights 1 --teacher_embedding_dimension ${TEACHER_DIM}"
OPTS+=" --teacher_model ${TEACHER} --teacher_tokenizer ${TEACHER_TOKENIZER}"
OPTS+=" --student_model ${MODEL} --student_tokenizer ${MODEL}"
OPTS+=" --student_device ${STUDENT_DEVICE} --teach_device ${TEACH_DEVICE}"
OPTS+=" --output_dir ${OUT} --seed ${SEED}"
OPTS+=" --student_model_type qwen --teacher_model_type ${TEACHER_TYPE}"
OPTS+=" ${PRECISION} ${WEIGHTS}"
if [[ -n "${DUMP_SPAN_WEIGHTS:-}" ]]; then
  OPTS+=" --dump_span_weights ${OUT}/span_weights.jsonl"
fi

"${PYTHON:-python}" run_distill_llm.py ${OPTS} >> "${OUT}/train.log" 2>&1
echo "trained ${STUDENT}/${METHOD} -> ${OUT}"
