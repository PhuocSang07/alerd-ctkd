import argparse
from arguments import Arguments
from teacher_llm import Teacher
from student import LLMModel, StudentCausalModel
from llm_train import Trainer, train
from types import SimpleNamespace

from transformers import HfArgumentParser
from huggingface_hub import login

import torch
import numpy as np
import random

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse(argv=None):
    hf_parser = HfArgumentParser(Arguments)
    args, remaining = hf_parser.parse_args_into_dataclasses(args=argv, return_remaining_strings=True)

    args: Arguments = args
    args.knowledge_distillation = True
    args.finetune_hidden_states = True
    args.finetune_embedding = True

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--teacher_sft", type=str, default=None)
    extra_parser.add_argument("--student_sft", type=str, default=None)
    extra_parser.add_argument("--student_model_type", type=str, default=None)
    extra_parser.add_argument("--teacher_model_type", type=str, default=None)

    return args, extra_parser.parse_args(remaining)


def build_trainer(args: Arguments, extras):
    set_seed(args.seed)

    token = args.hf_token or None
    if token:
        login(token)

    load_model_kwargs = {'torch_dtype': getattr(torch, args.teacher_dtype),
                         'device_map': args.teach_device,
                         'trust_remote_code': True,
                         'attn_implementation': 'sdpa',
                         'token': token}

    teacher_model = Teacher(model_name=args.teacher_model,
                            load_model_kwargs=load_model_kwargs,
                            export_hidden_state_layers=args.teacher_layers_mapping,
                            weight_pooling=args.span_weight_pooling,
                            span_weight=args.span_loss_weight,
                            sft_path=extras.teacher_sft)

    load_student_model_kwargs = {'device_map': args.student_device,
                                 'torch_dtype': getattr(torch, args.student_dtype),
                                 'attn_implementation': 'sdpa',
                                 'token': token}

    lora_config = None
    if args.use_lora:
        lora_config = SimpleNamespace(lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                      lora_dropout=args.lora_dropout, lora_target_modules=LORA_TARGETS)

    llm_model = LLMModel(model_name=args.student_model,
                         load_model_kwargs=load_student_model_kwargs,
                         hidden_layer_fineturn=args.student_encoder_layers_finetuned,
                         weight_pooling=args.span_weight_pooling,
                         span_weight=args.span_loss_weight,
                         lora_conf=lora_config, sft_path=extras.student_sft,
                         gradient_checkpointing=args.gradient_checkpointing)

    student_model = StudentCausalModel(llm_model, model_path=args.student_model,
                                       n_encoder_finetuned=args.n_encoder_finetuned,
                                       teacher_hidden_size=args.teacher_embedding_dimension,
                                       finetune_embedding=args.finetune_embedding,
                                       orthogonal=args.orthogonal)

    trainer = Trainer(student_model, extras.student_model_type, extras.teacher_model_type, args=args,
                      teacher_model=teacher_model, hidden_loss_weights=args.hidden_loss_weights)

    return trainer


def main(argv=None):
    args, extras = parse(argv)
    train(args, build_trainer(args, extras))


if __name__ == "__main__":
    main()
