from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import nn, Tensor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from layer_capture import LayerCapture
from utils import get_span_hidden_states

import os

import logging
logger = logging.getLogger(__name__)



@dataclass
class StudentOutput(ModelOutput):
    logits: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    hidden_states: Any = None
    span_weights: Any = None


class LLMModel(torch.nn.Module):
    def __init__(self, model_name, load_model_kwargs = {}, hidden_layer_fineturn=[23],
                 weight_pooling=True, span_weight=True, lora_conf=None, sft_path=None,
                 gradient_checkpointing=False):
        super().__init__()

        self.hidden_layer_fineturn = hidden_layer_fineturn
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight
        self.lora_config = lora_conf

        load_model_kwargs = {k: v for k, v in load_model_kwargs.items()
                             if k not in ('output_hidden_states', 'output_attentions')}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)

        if sft_path is not None:
            print("Loading adapter for student")
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()

        if lora_conf is not None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_conf.lora_rank,
                lora_alpha=lora_conf.lora_alpha,
                lora_dropout=lora_conf.lora_dropout,
                target_modules=lora_conf.lora_target_modules
            )
            self.model = get_peft_model(self.model, lora_config).to(self.model.device)
            self.model.print_trainable_parameters()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if lora_conf is not None:
                self.model.enable_input_require_grads()

        self.device = self.model.device

        self.capture = LayerCapture(self.model, hidden_layer_fineturn)

    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)
        inputs.pop('example_ids', None)

        self.capture.begin(inputs['attention_mask'])
        outputs = self.model(**inputs, return_dict=True, use_cache=False)
        self.capture.end()

        if not self.training:
            return StudentOutput(logits=outputs.logits)

        hidden_states, span_weights = None, None
        if safe_idx is not None:
            hidden_states, span_weights = get_span_hidden_states(
                self.capture.hidden_states(), self.capture.rows, safe_idx, pooler_mask,
                self.hidden_layer_fineturn, self.weight_pooling, self.span_weight)

        return StudentOutput(
            logits=outputs.logits,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def lm_head_parameters(self):
        """(weight, bias) of the output projection, also through a PEFT wrapper."""
        head = self.model.get_output_embeddings()
        return head.weight, getattr(head, 'bias', None)

    def save(self, output_dir: str):
        """Save an HF checkpoint (bf16). LoRA is merged into the base weights, which ends its training."""
        model = self.model
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
        model.to(torch.bfloat16).save_pretrained(output_dir, safe_serialization=True)

    def get_config(self):
        return self.model.config

class StudentCausalModel(torch.nn.Module):
    def __init__(self, model:LLMModel, model_path, n_encoder_finetuned,
                 teacher_hidden_size=-1, finetune_embedding=False, orthogonal=True):
        super().__init__()
        self.model = model

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        self.device = self.model.device

        self.proj_hidden_layers = None

        if teacher_hidden_size > 0:
            proj_list = []
            for i in range(len(self.model.hidden_layer_fineturn)):
                W = nn.Parameter(torch.empty(self.model.model.config.hidden_size, teacher_hidden_size))
                if orthogonal:
                    nn.init.orthogonal_(W)
                else:
                    nn.init.xavier_uniform_(W)
                proj_list.append(W)

            self.proj_hidden_layers = nn.ParameterList(proj_list)

            self.proj_embeddings = nn.Parameter(torch.empty(self.model.get_config().hidden_size, teacher_hidden_size))
            if orthogonal:
                nn.init.orthogonal_(self.proj_embeddings)
            else:
                nn.init.xavier_uniform_(self.proj_embeddings)

            hidden_weight_path = os.path.join(model_path, 'proj_hidden_layers.pt')
            if os.path.exists(hidden_weight_path):
                self.proj_hidden_layers = torch.load(hidden_weight_path, weights_only=False)

            if os.path.exists(os.path.join(model_path, 'proj_embeddings.pt')):
                self.proj_embeddings = torch.load(os.path.join(model_path, 'proj_embeddings.pt'),
                                                  weights_only=False)

            self.proj_hidden_layers.to(self.device)
            self.proj_embeddings = nn.Parameter(self.proj_embeddings.to(self.device))

    def decode(self, inputs) -> StudentOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(inputs)

        if outputs.hidden_states is not None and self.proj_hidden_layers is not None:
            outputs.embeddings = outputs.hidden_states[-1]
            outputs.hidden_states = torch.stack([outputs.hidden_states[i] @ proj_layer
                                                 for i, proj_layer in enumerate(self.proj_hidden_layers)])

        return outputs

    def save(self, path: str):
        """HF checkpoint in `path`; the KD projections go to `path/kd`, outside what eval loads."""
        self.model.save(path)
        if self.proj_hidden_layers is not None:
            os.makedirs(os.path.join(path, 'kd'), exist_ok=True)
            torch.save(self.proj_hidden_layers, os.path.join(path, 'kd', 'proj_hidden_layers.pt'))
            torch.save(self.proj_embeddings, os.path.join(path, 'kd', 'proj_embeddings.pt'))
