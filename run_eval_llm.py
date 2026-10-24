import argparse
from arguments import Arguments

from evaluator import Evaluator

from transformers import HfArgumentParser
from huggingface_hub import login

import torch
import json
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    hf_parser = HfArgumentParser(Arguments)
    args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args: Arguments = args
    args.knowledge_distillation = True
    args.finetune_hidden_states = True
    args.output_attentions = True
    args.weight_decay = 0.01
    args.warmup_ratio = 0.1
    args.finetune_embedding = True


    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    extra_parser.add_argument("--teacher_sft", type=str, default=None)
    extra_parser.add_argument("--student_sft", type=str, default=None)
    extra_parser.add_argument("--student_model_type", type=str, default=None)
    extra_parser.add_argument("--teacher_model_type", type=str, default=None)
    extra_parser.add_argument("--use_lora", type=bool, default=False)
    extra_parser.add_argument("--grad_accum_steps", type=int, default=1)

    extras = extra_parser.parse_args(remaining)

    set_seed(extras.seed)

    login(args.hf_token)

    evaluator = Evaluator(
        tokenizer_path=args.student_tokenizer,
        model_path=args.student_model,
        sft_lora=None,
        distilled_lora=args.output_dir,
        seeds=[10, 20, 30, 40, 50]
    )

    # evaluator.model = trainer.student.model.model

    benchmark_configs = {'dolly': './data/dolly/valid.jsonl',
                        'self_instruct': './data/self-inst/valid.jsonl',
                        'vicuna': './data/vicuna/valid.jsonl',
                        'sni': './data/sinst/11_/valid.jsonl'
                        }

    results = evaluator.evaluate_multiple_benchmarks(
        benchmark_configs=benchmark_configs,
        batch_size=32,
        max_seq_length=256,
        max_new_tokens=512
    )

    with open(args.output_dir + "/eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    result = evaluator.evaluate_benchmark_dataset(
            dataset_path='./data/dialog/valid.jsonl',
            dataset_name='dialog', batch_size=32,
            max_seq_length=512, max_new_tokens=384)
    
    dialog_result = {"rouge_l_f1": result, "status": "success"}
    with open(args.output_dir + "/dialog_result_eval.json", "w", encoding="utf-8") as f:
        json.dump(dialog_result, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()