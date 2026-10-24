from arguments import Arguments
from teacher_llm import Teacher, TeacherOutput
from student import StudentCausalModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator
from kd_losses import make_weights, cosine_loss, geometry_loss, kl_loss
from span_weights import SpanWeightProvider

from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
import json
import math
import os

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_tokenizer(model_type, path, kwargs):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    # keep the tokenizer's own EOS (the end-of-turn token of the chat template); pad only when it has none
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # span indices start at the prompt length, which only holds with right padding
    tokenizer.padding_side = 'right'
    return tokenizer


def get_token_mapping(s_tokenizer, t_tokenizer, device):
    t_vocab = t_tokenizer.get_vocab()
    s_vocab = s_tokenizer.get_vocab()
    t_id_mapping = []
    s_id_mapping = []
    for s_token, s_token_id in s_vocab.items():
        if s_token in t_vocab:
            s_id_mapping.append(s_token_id)
            t_id_mapping.append(t_vocab[s_token])

    return torch.tensor(s_id_mapping, device=device), torch.tensor(t_id_mapping, device=device)


def cosine_schedule(optimizer, warmup_steps, total_steps, min_ratio):
    """Linear warm-up, then cosine decay of the learning rate to min_ratio times its peak."""
    def factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, student: StudentCausalModel, student_model_type: str, teacher_model_type: str,
                 args: Arguments, teacher_model: Teacher = None, hidden_loss_weights = None):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature

        self.step = 0
        self.last_stats = {}

        n_layers = len(args.student_encoder_layers_finetuned)
        hidden_loss_weights = hidden_loss_weights or [1.0] * n_layers
        if len(hidden_loss_weights) != n_layers:
            raise ValueError("hidden_loss_weights needs one weight per distilled layer")
        self.hidden_loss_weights = [w / sum(hidden_loss_weights) for w in hidden_loss_weights]

        self.span_provider = None
        if args.span_weight_source != 'attention':
            self.span_provider = SpanWeightProvider(
                args.span_weight_source, args.spans_file, args.train_data, gain_file=args.gain_file,
                predictability_file=args.predictability_file, tau=args.alloc_tau, clip=args.alloc_clip,
                lam=args.alloc_lambda, normalization=args.alloc_norm, answer_only_unit=args.answer_only_unit,
                seed=args.seed)

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, student_model_type, teacher_model_type)

        self.s_id_mapping, self.t_id_mapping = get_token_mapping(self.student_tokenizer,
                                                                 self.teacher_tokenizer,
                                                                 device=self.student.device)

        # only the shared-vocabulary rows of the teacher's lm_head are ever used
        head = self.teacher_model.lm_head
        self.teacher_head_weight = head.weight[self.t_id_mapping.to(head.weight.device)].detach().to(self.student.device)
        bias = getattr(head, 'bias', None)
        self.teacher_head_bias = None if bias is None else bias[self.t_id_mapping.to(bias.device)].detach().to(self.student.device)

    def get_data_loader(self, args: Arguments, student_model_type: str, teacher_model_type: str):
        self.student_tokenizer = load_tokenizer(student_model_type, args.student_tokenizer,
                                                args.load_student_tokenizer_kwargs)
        self.teacher_tokenizer = load_tokenizer(teacher_model_type, args.teacher_tokenizer,
                                                args.load_teacher_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.student_tokenizer, self.teacher_tokenizer)

        train_collate = LLMDataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                       do_train=True, max_len = args.max_len,
                                       pad_to_multiple_of = args.pad_to_multiple_of,
                                       return_tensors = 'pt', padding = True,
                                       span_provider=self.span_provider)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None


    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)

        outputs.hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)
        outputs.span_weights = outputs.span_weights.to(self.student.device, non_blocking=True)
        outputs.pooler_mask = outputs.pooler_mask.to(self.student.device, non_blocking=True)

        return outputs

    def knowledge_distillation_loss(self, student_outputs: StudentOutput, teacher_outputs: TeacherOutput,
                                    omega=None):
        valid = teacher_outputs.pooler_mask.any(dim=-1)  # (B, N) real spans
        mass = teacher_outputs.span_weights.squeeze(-1)  # (L, B, N)
        layer_w, pair_w, kl_w = make_weights(mass, valid, self.args.p, omega)

        terms = {}
        if self.args.span_loss:
            terms['cosine'] = cosine_loss(student_outputs.hidden_states, teacher_outputs.hidden_states,
                                          layer_w, self.hidden_loss_weights)
        teacher_last = teacher_outputs.hidden_states[-1]
        terms['geometry'] = self.args.geom_loss_weight * geometry_loss(student_outputs.embeddings, teacher_last, pair_w)

        s_weight, s_bias = self.student.model.lm_head_parameters()
        s_logits = F.linear(student_outputs.embeddings, s_weight[self.s_id_mapping],
                            None if s_bias is None else s_bias[self.s_id_mapping])
        t_logits = F.linear(teacher_last.to(self.teacher_head_weight.dtype), self.teacher_head_weight,
                            self.teacher_head_bias)
        terms['kl'] = kl_loss(s_logits, t_logits, kl_w, self.temperature)

        self._record_weights(layer_w[-1], valid, omega)
        return sum(terms.values()), terms

    def _record_weights(self, weights, valid, omega):
        """Percentiles of N * weight over real spans (1 = uniform) and the mean budget sum(weight) per sample."""
        with torch.no_grad():
            n = valid.sum(-1, keepdim=True).clamp(min=1)
            scaled = (weights * n)[valid].float()
            q = torch.quantile(scaled, torch.tensor([0.1, 0.5, 0.9], device=scaled.device))
            self.last_stats.update({'w_p10': q[0].item(), 'w_p50': q[1].item(), 'w_p90': q[2].item(),
                                    'budget': (weights * valid).sum(-1).mean().item()})

    def compute_loss(self, student_inputs, labels, teacher_outputs = None):
        omega = student_inputs.pop('span_weights_alloc', None)
        example_ids = student_inputs.pop('example_ids', None)
        student_outputs = self.student.decode(student_inputs)

        logits = student_outputs.logits
        # logits at position t predict labels[:, t + 1]; labels carry the response and the EOS
        hard_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100)

        kd_loss, terms = 0, {}

        if self.args.knowledge_distillation and teacher_outputs is not None:
            kd_loss, terms = self.knowledge_distillation_loss(student_outputs, teacher_outputs,
                                                              None if omega is None else omega.to(logits.device))
            if self.args.dump_span_weights:
                self._dump(example_ids, teacher_outputs, omega)

        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        self.step += 1
        self.last_stats.update({k: float(v) for k, v in terms.items()})

        return loss, hard_loss

    def _dump(self, example_ids, teacher_outputs, omega):
        mass = teacher_outputs.span_weights[-1].squeeze(-1)
        valid = teacher_outputs.pooler_mask.any(dim=-1)
        with open(self.args.dump_span_weights, 'a', encoding='utf-8') as handle:
            for b, rid in enumerate(example_ids.tolist()):
                n = int(valid[b].sum())
                row = {'id': rid, 'attention': (mass[b, :n] / mass[b, :n].sum()).tolist()}
                if omega is not None:
                    row['omega'] = omega[b, :n].tolist()
                handle.write(json.dumps(row) + '\n')


def train(args: Arguments, trainer: Trainer, grad_accum_steps=None):
    grad_accum_steps = grad_accum_steps or args.grad_accum_steps
    trainer.student.train()
    trainer.student.model.train()

    train_loader = trainer.train_loader
    device = trainer.student.device

    # fp32 master weights; autocast runs the matmuls in bf16, so no loss scaling is needed
    optimizer = optim.AdamW([p for p in trainer.student.model.parameters() if p.requires_grad],
                            lr=args.learning_rate, weight_decay=args.weight_decay)
    if trainer.student.proj_hidden_layers is not None:
        optimizer.add_param_group({"params": trainer.student.proj_hidden_layers.parameters(),
                                   "lr": args.proj_learning_rate, "weight_decay": 0.0})
        optimizer.add_param_group({"params": [trainer.student.proj_embeddings],
                                   "lr": args.proj_learning_rate, "weight_decay": 0.0})

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_training_steps = steps_per_epoch * args.num_train_epochs
    scheduler = cosine_schedule(optimizer, int(total_training_steps * args.warmup_ratio), total_training_steps,
                                args.min_learning_rate / args.learning_rate)

    stats_path = os.path.join(args.output_dir, 'train_stats.jsonl')
    os.makedirs(args.output_dir, exist_ok=True)
    window = []

    def optimizer_step():
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        keys = {k for row in window for k in row}
        row = {k: sum(r.get(k, 0.0) for r in window) / len(window) for k in keys}
        row['step'] = trainer.step
        row['lr'] = scheduler.get_last_lr()[0]
        with open(stats_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        window.clear()

    for epoch in range(args.num_train_epochs):
        p_bar = tqdm(train_loader, total=len(train_loader))
        loss_total = 0
        micro = 0

        for batch in p_bar:
            student_inputs, teacher_inputs, labels = batch

            teacher_outputs = trainer.get_teacher_eval(teacher_inputs)

            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, student_loss = trainer.compute_loss(student_inputs, labels, teacher_outputs)

            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {trainer.step}: {trainer.last_stats}')

            (loss / grad_accum_steps).backward()
            window.append({'loss': loss.item(), 'ce': student_loss.item(), **trainer.last_stats})
            micro += 1

            if micro % grad_accum_steps == 0:
                optimizer_step()

            loss_total += loss.item()
            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G' if torch.cuda.is_available() else 'cpu'
            p_bar.set_description(f'{epoch + 1}/{args.num_train_epochs} {memory} loss {loss_total / micro:.5g}')

        if micro % grad_accum_steps:  # flush the last partial accumulation window
            optimizer_step()

    trainer.student.save(args.output_dir)
    trainer.student_tokenizer.save_pretrained(args.output_dir)
