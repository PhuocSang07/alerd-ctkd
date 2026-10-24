from typing import Optional, Any
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from peft import PeftModel

from layer_capture import LayerCapture
from utils import get_span_hidden_states

import logging
logger = logging.getLogger(__name__)


@dataclass
class TeacherOutput(ModelOutput):
    hidden_states: Optional[Tensor] = None   # (L, B, N, D) span-pooled states of the exported layers
    span_weights: Optional[Tensor] = None    # (L, B, N, 1) attention mass of each span
    pooler_mask: Any = None
    pooler_idx: Any = None
    attention_mask: Any = None


class Teacher:
    """Frozen teacher that is only teacher-forced: it returns span-level states, never full-sequence logits.

    Works for any Llama-style decoder (Qwen2/Qwen3/Llama/Mistral). Attention runs fused for every layer; the
    exported layers carry hooks that store their output and last-token attention row (layer_capture).
    """

    def __init__(self, model_name, load_model_kwargs, export_hidden_state_layers,
                 weight_pooling=True, span_weight=True, sft_path=None):
        print('Teacher loading model ...')
        load_model_kwargs = {k: v for k, v in load_model_kwargs.items()
                             if k not in ('output_hidden_states', 'output_attentions')}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)
        if sft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()
        self.model = self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = self.model.device
        self.export_hidden_state_layers = export_hidden_state_layers
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight
        self.capture = LayerCapture(self.model, export_hidden_state_layers)

    @property
    def lm_head(self):
        return self.model.get_output_embeddings()

    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx')
        pooler_mask = inputs.pop('pooler_mask')

        self.capture.begin(inputs['attention_mask'])
        with torch.no_grad():
            self.model.base_model(**inputs, use_cache=False)
        self.capture.end()

        hidden_states, span_weights = get_span_hidden_states(
            self.capture.hidden_states(), self.capture.rows, safe_idx, pooler_mask,
            self.export_hidden_state_layers, self.weight_pooling, self.span_weight)

        return TeacherOutput(
            hidden_states=hidden_states,
            span_weights=span_weights,
            pooler_mask=pooler_mask,
            pooler_idx=safe_idx,
            attention_mask=inputs['attention_mask'],
        )
