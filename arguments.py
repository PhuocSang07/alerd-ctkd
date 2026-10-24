from dataclasses import dataclass, field
from typing import List, Optional, Any
import os



@dataclass
class Arguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to training data"}
    )
    val_data: str = field(
        default=None, metadata={"help": "Path to validation data"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Path to test data"}
    )

    num_labels: int = field(default=2, metadata={"help": "Number of labels"})

    batch_size: int = field(default=1)
    val_batch_size: int = field(default=32)


    max_len: int = field(
        default=2500,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    pad_to_multiple_of: int = field(default=1, metadata={"help": ""})

    temperature: Optional[float] = field(default=2.0)
    distill_temperature: Optional[float] = field(default=2.0)
 
    knowledge_distillation: bool = field(default=True, metadata={"help": "Use knowledge distillation"})
    finetune_hidden_states: bool = field(default=True)
    output_attentions: bool = field(default=True)
    
    teach_device: str = field(default='cuda:1')
    student_device: str = field(default='cuda:0')

    num_train_epochs: int = field(default=3)

    learning_rate: float = field(default=5e-5)
    min_learning_rate: float = field(default=1e-5, metadata={"help": "cosine decays to this value"})
    proj_learning_rate: float = field(default=5e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)

    geom_loss_weight: float = field(default=50)
    hard_label_loss_weight: float = field(default=1.0)

    teacher_layers_mapping: List[int] = field(default=list)
    student_encoder_layers_finetuned: List[int] = field(default=list)
    n_encoder_finetuned: int = field(default=6)
    finetune_embedding: bool = field(default=False)

    orthogonal: bool = field(default=True)
    span_loss: bool = field(default=True)

    span_weight_pooling: bool = field(default=True)
    span_loss_weight: bool = field(default=True)

    p: float = field(default=1.0)

    hidden_loss_weights: List[float] = field(default=None)
    teacher_embedding_dimension: int = field(default=1024)


    output_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the final model"})

    teacher_model: str = field(default='')
    teacher_tokenizer: str = field(default='')
    student_model: str = field(default='google-bert/bert-base-uncased')
    student_tokenizer: str = field(default='google-bert/bert-base-uncased')
    hf_token: str = field(default_factory=lambda: os.environ.get('HF_TOKEN', ''),
                          metadata={"help": "defaults to the HF_TOKEN environment variable"})

    # training recipe (ALERD Appendix A): effective batch = batch_size * grad_accum_steps
    grad_accum_steps: int = field(default=32)
    use_lora: bool = field(default=False, metadata={"help": "LoRA on every attention and MLP projection"})
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    student_dtype: str = field(default='float32', metadata={"help": "float32 master weights (full fine-tuning) or bfloat16 base for LoRA"})
    teacher_dtype: str = field(default='bfloat16')
    gradient_checkpointing: bool = field(default=True)
    seed: int = field(default=42)

    # span weights of the KD losses
    span_weight_source: str = field(
        default='attention',
        metadata={"help": "attention (SRA) | uniform | alerd | random | predictability-easy | predictability-hard"})
    spans_file: str = field(default=None, metadata={"help": "spans.jsonl from span_index.py (needed unless source=attention)"})
    gain_file: str = field(default=None, metadata={"help": "answer-gain json from sgl.signals.answer_gain"})
    predictability_file: str = field(default=None, metadata={"help": "per-step mean log-prob json from score_predictability.py"})
    alloc_tau: float = field(default=1.0)
    alloc_clip: float = field(default=2.0)
    alloc_lambda: float = field(default=0.5)
    alloc_norm: str = field(default='per-trace', metadata={"help": "per-trace | corpus | none"})
    answer_only_unit: bool = field(default=False, metadata={"help": "keep steps that restate the answer at weight 1"})
    dump_span_weights: str = field(default=None, metadata={"help": "write per-span attention and omega weights here (jsonl)"})

    load_student_tokenizer_kwargs: dict = field(default_factory=dict)
    load_teacher_tokenizer_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
            
        if self.span_weight_source not in ('attention', 'uniform', 'alerd', 'random',
                                           'predictability-easy', 'predictability-hard'):
            raise ValueError(f"unknown span_weight_source {self.span_weight_source!r}")
        if self.span_weight_source != 'attention' and not self.spans_file:
            raise ValueError("span_weight_source other than 'attention' needs --spans_file")
        if self.alloc_norm not in ('per-trace', 'corpus', 'none'):
            raise ValueError(f"unknown alloc_norm {self.alloc_norm!r}")
        if len(self.teacher_layers_mapping) != len(self.student_encoder_layers_finetuned):
            raise ValueError("teacher_layers_mapping and student_encoder_layers_finetuned should have the same length")

