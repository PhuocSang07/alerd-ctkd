# SRA: Span Representation Alignment for Large Language Model Distillation

This repository provides the implementation of **SRA**, a framework for cross-tokenizer knowledge distillation between large and small language models.  
It enables span-level alignment to transfer knowledge effectively across heterogeneous tokenizers.

---

## 1. Environment Setup

Make sure you are using **Python ≥ 3.9** and **PyTorch ≥ 2.6** with CUDA enabled.

Install all dependencies:
```
pip install -r requirements.txt
```

## 2. Data Format
Training and evaluation data should be provided in .jsonl format, where each line follows this structure:

```
{
  "instruction": "...",
  "prompt": "Below is an instruction...\n\n### Instruction: ... \n\n### Response:\n",
  "input": "...",
  "output": "..."
}


```

## 3. Run Code

- Create a bash script (e.g., run_sra.sh) with the following content:

```
#! /bin/bash

SEED=$1

OUTPUT_DIR=<path_to_output_dir>

mkdir -p ${OUTPUT_DIR}

OPTS=""

# data .jsonl 
OPTS+=" --train_data <path_to_train_data>"
OPTS+=" --val_data <path_to_val_data>"
OPTS+=" --test_data <path_to_test_data>"

# training
OPTS+=" --num_train_epochs 10"
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
OPTS+=" --span_loss True"
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
OPTS+=" --hf_token <hf_token>"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --teacher_sft <path_to_sft_model>"
OPTS+=" --student_model_type tinyllama"
OPTS+=" --teacher_model_type mistral"
OPTS+=" --use_lora True"
OPTS+=" --grad_accum_steps 4"

# ==== run code====
python run_distill_llm.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1

```


- To run with a fixed random seed (e.g., 42):

```
bash run_sra.sh 42
```

- Notes:

    - Replace all placeholders (<path_to_...>, <hf_token>) with your actual paths and credentials.

    - Logs and checkpoints will be automatically saved in ${OUTPUT_DIR}.
    
    - Add path to eval data in run_distill_llm.py