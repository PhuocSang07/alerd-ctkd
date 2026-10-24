#! /bin/bash

SEED=$1

# ==== Định nghĩa các biến ====
BASE_PATH=.
OUTPUT_DIR="${BASE_PATH}/outputs/tiny_llama/seed-${SEED}"
CKPT_NAME="mistral-tiny-llama-checkpoint"

mkdir -p ${OUTPUT_DIR}

# Gom tham số vào OPTS
OPTS=""
# data
OPTS+=" --train_data ${BASE_PATH}/data/llm/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/llm/dolly/valid.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/llm/vicuna/valid.jsonl"

# training
OPTS+=" --num_train_epochs 15"
OPTS+=" --batch_size 4"
OPTS+=" --val_batch_size 32"
OPTS+=" --learning_rate 1e-3"
OPTS+=" --max_len 320"
OPTS+=" --pad_to_multiple_of 1"

# devices
OPTS+=" --teach_device auto"
OPTS+=" --student_device auto"

# loss
OPTS+=" --hard_label_loss_weight 0.5"
OPTS+=" --orthogonal True"
OPTS+=" --span_loss True"
OPTS+=" --der_loss False"
OPTS+=" --span_weight_pooling True"
OPTS+=" --span_loss_weight True"
OPTS+=" --p 1.0"

OPTS+=" --teacher_layers_mapping 32"
OPTS+=" --student_encoder_layers_finetuned 22"
OPTS+=" --n_encoder_finetuned 22"
OPTS+=" --hidden_loss_weights 1"


# models
OPTS+=" --teacher_embedding_dimension 4096"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --teacher_model mistralai/Mistral-7B-v0.1"
OPTS+=" --teacher_tokenizer mistralai/Mistral-7B-v0.1"
OPTS+=" --student_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OPTS+=" --student_tokenizer TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# hf token
OPTS+=" --hf_token hf_***"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --teacher_sft VoCuc/Mistral7B_Dolly_SFT"
OPTS+=" --student_model_type tinyllama"
OPTS+=" --teacher_model_type mistral"
OPTS+=" --use_lora True"
OPTS+=" --grad_accum_steps 4"

# ==== Gọi Python ====
python run_distill_llm.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1
