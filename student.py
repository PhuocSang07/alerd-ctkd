from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import nn, Tensor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import get_span_hidden_states, get_span_hidden_states_custom
import re
    
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
                 weight_pooling=True, span_weight=True, lora_conf=None, sft_path=None):
        super().__init__()

        self.hidden_layer_fineturn = hidden_layer_fineturn
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight
        self.lora_config = lora_conf

        self.saved_hidden_states = []

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config
        
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

        self.device = self.model.device

        self._register_hooks()

    def _register_hooks(self):
        
        def save_hidden_states_hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.saved_hidden_states.append(hidden_state)

        for name, module in self.model.named_modules():
            if re.match(r".*\.layers?\.\d+$", name):
                module.register_forward_hook(save_hidden_states_hook)
            elif re.match(r".*\.h?\.\d+$", name):
                module.register_forward_hook(save_hidden_states_hook)
            


    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)
        inputs.pop('example_ids', None)

        self.saved_hidden_states.clear()
        self.saved_hidden_states.append(None)

        outputs = self.model(**inputs, output_attentions=True, return_dict=True, use_cache=False)

        if not self.training:
            return StudentOutput(logits=None)
        
        if len(self.saved_hidden_states) > 0:
            hidden_states = tuple(self.saved_hidden_states)
        else:
            hidden_states = None
        
        attentions =  outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), 
                                     device=inputs['input_ids'].device)

        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, 
                                                                      pooler_mask, inputs['attention_mask'],
                                                                      self.hidden_layer_fineturn,
                                                                      self.weight_pooling, self.span_weight, 
                                                                      is_causal=True)
            
        self.saved_hidden_states.clear() 

        return StudentOutput(
            logits=outputs.logits,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())
    
    def get_config(self):
        return self.model.config
    
class StudentCausalModel(torch.nn.Module):
    def __init__(self, model:LLMModel, model_path, n_encoder_finetuned, 
                 teacher_hidden_size=-1, finetune_embedding=False, orthogonal=True):
        super().__init__()
        self.model = model

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print('model output_attentions:', model.get_config().output_attentions)
        print('model output_attentions:', model.get_config().output_hidden_states)
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
            hidden_states = []
            outputs.embeddings = outputs.hidden_states[-1]
            for i, proj_layer in enumerate(self.proj_hidden_layers):
                hidden_states.append(outputs.hidden_states[i] @ proj_layer)
                
            outputs.hidden_states = hidden_states

        return outputs

    def save(self, path: str):
        self.model.save(path)
        if self.proj_hidden_layers is not None:
            torch.save(self.proj_hidden_layers, os.path.join(path, 'proj_hidden_layers.pt'))

        if self.proj_embeddings is not None:
            torch.save(self.proj_embeddings, os.path.join(path, 'proj_embeddings.pt'))
