from arguments import Arguments
from teacher_llm import Teacher, TeacherOutput
from student import StudentCausalModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator
from loss import cosine_token_weight_loss

from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import torch

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from evaluator import Evaluator


def load_tokenizer(model_type, path, kwargs):        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "qwen":
        # tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('tokenizer unknow')
    
    return tokenizer

from typing import Type
from torch.utils.data import DataLoader, Dataset
from torch import nn


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

class Trainer:
    def __init__(self, student: StudentCausalModel, student_model_type: str, teacher_model_type: str,
                 args: Arguments, teacher_model: Teacher = None,
                 hidden_loss_weights = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature

        self.step = 0

        sum_hidden_loss_weights = sum(hidden_loss_weights)
        self.hidden_loss_weights = [w / sum_hidden_loss_weights for w in hidden_loss_weights]

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, student_model_type, teacher_model_type)

        self.s_vocab_size = self.student.model.model.config.vocab_size
        self.student_loss_function = self.student.model.model.loss_function

        
        self.teacher_lm_head = nn.Linear(self.teacher_model.model.lm_head.in_features,
                                         self.teacher_model.model.lm_head.out_features,
                                         bias=(self.teacher_model.model.lm_head.bias is not None)
                                        ).to(self.student.device)
        self.teacher_lm_head.load_state_dict(self.teacher_model.model.lm_head.state_dict())
        for p in self.teacher_lm_head.parameters():
            p.requires_grad = False

        self.s_id_mapping, self.t_id_mapping = get_token_mapping(self.student_tokenizer, 
                                                                 self.teacher_tokenizer, 
                                                                 device=self.student.device)

    def get_data_loader(self, args: Arguments, student_model_type: str, teacher_model_type: str):
        self.student_tokenizer = load_tokenizer(student_model_type, args.student_tokenizer, 
                                                args.load_student_tokenizer_kwargs)
        self.teacher_tokenizer = load_tokenizer(teacher_model_type, args.teacher_tokenizer, 
                                                args.load_teacher_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.student_tokenizer, 
                                   self.teacher_tokenizer, args.max_len // 2)

        train_collate = LLMDataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                       do_train=True, max_len = args.max_len,
                                       pad_to_multiple_of = args.pad_to_multiple_of,
                                       return_tensors = 'pt', padding = True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None


    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)
  
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)
            
        if outputs.span_weights is not None:
            outputs.span_weights=outputs.span_weights.to(self.student.device, non_blocking=True)

        return outputs

    def soft_label_distill_loss(self, student_logits, teacher_logits, distill_temperature = 2.0):
        
        student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

        mask = (student_logits.abs().sum(dim=-1) != 0).float()
        loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
        loss = (loss * mask).sum() / student_logits.size(0)

        return loss

    def knowledge_distillation_loss(self, student_outputs: StudentOutput,
                                    teacher_outputs: TeacherOutput = None):
        kd_loss = 0
        temp_loss = torch.tensor(0)

        if teacher_outputs is not None:
            if teacher_outputs.hidden_states is not None:
                span_loss = 0
                n_layer = teacher_outputs.hidden_states.size(0)
                span_weights = teacher_outputs.span_weights.squeeze(-1)
                _, B, N = span_weights.size()

                mask = span_weights[-1].bool()  # [B, N]

                span_weights = span_weights ** self.args.p
                span_weights = span_weights / span_weights.sum(-1, keepdim=True)

                pair_weights = span_weights[-1].unsqueeze(2) * span_weights[-1].unsqueeze(1)
                mask = torch.eye(N, device=pair_weights.device).bool()  # (N, N)
                pair_weights[:, mask] = 0.0
                pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

                
                span_weights = span_weights.unsqueeze(-1)
                if self.args.span_loss:
                    for i in range(n_layer):
                        s_hidden = student_outputs.hidden_states[i]
                        t_didden = teacher_outputs.hidden_states[i]
                        span_w = span_weights[i]

                        state_loss = cosine_token_weight_loss(s_hidden, t_didden, span_w)
            
                        span_loss += self.hidden_loss_weights[i] * state_loss

                        if torch.isnan(span_loss):
                            print('span_loss nan')
                

                kd_loss += 1 * span_loss

                s_hidden = F.normalize(student_outputs.embeddings, dim=-1, eps=1e-5)
                t_hidden = F.normalize(teacher_outputs.hidden_states[n_layer - 1], dim=-1, eps=1e-5)
                
                student_scores = torch.matmul(s_hidden, s_hidden.transpose(-1, -2))
                teacher_scores = torch.matmul(t_hidden, t_hidden.transpose(-1, -2))
                score_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
                score_loss = (score_loss * pair_weights).sum() / B
    
                kd_loss += self.args.geom_loss_weight * score_loss


                s_logits = self.student.model.model.lm_head(student_outputs.embeddings)
                t_logits = self.teacher_lm_head(teacher_outputs.hidden_states[n_layer - 1])
                
                s_map_logits = s_logits[:, :, self.s_id_mapping]
                t_map_logits = t_logits[:, :, self.t_id_mapping]
                kd_loss += self.soft_label_distill_loss(s_map_logits, t_map_logits, self.temperature)

        return kd_loss, temp_loss.item()

    
    def compute_loss(self, student_inputs, labels, teacher_outputs = None):
        student_outputs = self.student.decode(student_inputs)
        
        hard_loss = self.student_loss_function(student_outputs.logits, 
                                               labels.view(-1), self.s_vocab_size)

        kd_loss, _t_loss_= 0, 0

        if self.args.knowledge_distillation and teacher_outputs is not None:
            kd_loss, _t_loss_ = self.knowledge_distillation_loss(student_outputs, teacher_outputs)

        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        self.step += 1

        return loss, hard_loss
    

def train(args: Arguments, trainer: Trainer, evaluator: Evaluator, grad_accum_steps=1):
    trainer.student.train()
    trainer.student.model.train()

    train_loader = trainer.train_loader

    optimizer = optim.AdamW(trainer.student.model.parameters(), lr=args.learning_rate)
    optimizer.add_param_group({"params": trainer.student.proj_hidden_layers.parameters(), "lr": 5e-4, "weight_decay": 0.0})
    optimizer.add_param_group({"params": [trainer.student.proj_embeddings], "lr": 5e-4, "weight_decay": 0.0})

    num_steps = len(train_loader) // grad_accum_steps + 1
    total_traning_steps = num_steps * args.num_train_epochs

    scaler = GradScaler()

    scheduler = get_scheduler(
        name='cosine_with_min_lr',
        optimizer=optimizer,
        num_warmup_steps=int(total_traning_steps * args.warmup_ratio),
        # num_warmup_steps=0,
        num_training_steps=total_traning_steps,
        scheduler_specific_kwargs={'min_lr': 5e-6}
    )

    best_result = 0

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
        p_bar = tqdm(train_loader, total=len(train_loader))
        loss_total = 0
        student_loss_total = 0
        step = 0

        for batch in p_bar:
            student_inputs, teacher_inputs, labels = batch

            teacher_outputs = trainer.get_teacher_eval(teacher_inputs)

            labels = labels.to(trainer.student.device)
            with autocast():
                loss, student_loss = trainer.compute_loss(student_inputs, labels, teacher_outputs)

            scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
            s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
                                                    loss_total / step, student_loss_total / step)
            p_bar.set_description(s)

            if torch.isnan(loss):
                break

        with torch.cuda.amp.autocast(dtype=torch.float16):
            evaluator.model = trainer.student.model.model
            dolly = evaluator.evaluate_benchmark_dataset(
                dataset_path=args.val_data,
                dataset_name='dolly', batch_size=16,
                max_seq_length=256, max_new_tokens=512)
        if dolly > best_result:
            best_result = dolly
            trainer.student.save(args.output_dir)
            
        trainer.student.save(args.output_dir + f'-epoch{epoch}')
            
        

    

