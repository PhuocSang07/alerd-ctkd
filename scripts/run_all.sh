#!/bin/bash

echo 'Start run training scripts, logs are being saved to ${OUTPUT_DIR}/train.log'


bash ./scripts/mistral_tinyllama.sh 42
bash ./scripts/qwen2_5_gpt2_1_5B.sh 42
bash ./scripts/qwen2_5_opt.sh 42

echo "=== All done ==="
