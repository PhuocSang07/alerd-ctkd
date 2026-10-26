r"""Off-policy distillation with per-span weights on the CE and on the KD loss (student and teacher share a tokenizer).

    python span_kd_train.py --train-data work/<student>/train-sra.jsonl --spans-file work/<student>/spans-<teacher>.jsonl \
        --gains-file work/<student>/span-gains.json --student-model <s> --teacher-model <t> --output-dir outputs/run \
        --ce-weights student --kd-weights diff

The recipe follows RIVAD's off-policy mode: LoRA student, loss (1 - beta) * CE + beta * KD, where KD is the forward KL on
the teacher's top-k logits with both distributions renormalised on that support (scaled by temperature^2). The student
and the teacher score the same canonical response; the response is cut into the word spans of word_spans.py.

Every span n of a sample gets a mass omega_n (sum_n omega_n = 1) and every token of the span an equal share of it:

    loss_CE = mean_samples sum_n omega^CE_n * mean_{t in span n} CE_t        (likewise for KD with omega^KD)

`--ce-weights` / `--kd-weights` pick omega: `token` (every token weighs the same), `uniform` (every span weighs the
same) or the budget-preserving answer-gain weights of ALERD Eq. 3-5 computed from the `student`, `teacher` or `diff`
(teacher minus student) span gains of span-gains.json. The EOS the student has to emit after the response joins the last
span, so the response is supervised until it ends.
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from allocation import trace_weights
from data_utils import text_hash
from llm_train import cosine_schedule
from token_gain import read_jsonl

STRUCTURAL_SOURCES = ("token", "uniform")
GAIN_SOURCES = ("student", "teacher", "diff")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def span_omega(source, sizes, gains, tau=1.0, clip=2.0, lam=0.5, normalization="per-trace"):
    """Mass of each span of one sample; sums to 1. `gains` is the sample's entry of span-gains.json."""
    n = len(sizes)
    if source == "token":
        return [size / sum(sizes) for size in sizes]
    if source == "uniform":
        return [1.0 / n] * n
    if source not in GAIN_SOURCES:
        raise ValueError(f"weight source must be one of {STRUCTURAL_SOURCES + GAIN_SOURCES}, got {source!r}")
    if len(gains[source]) != n:
        raise ValueError(f"{len(gains[source])} {source} gains for {n} spans")
    weights = trace_weights(gains[source], [1] * n, tau, lam, clip, 1e-8, normalization)
    return [w / n for w in weights]


class SpanKDDataset(Dataset):
    """The samples of the span index with their response spans and the masses of both losses."""

    def __init__(self, train_data, spans_file, gains_file, ce_source, kd_source, alloc=None):
        alloc = alloc or {}
        needs_gains = any(s in GAIN_SOURCES for s in (ce_source, kd_source))
        if needs_gains and not gains_file:
            raise ValueError(f"weights {ce_source!r}/{kd_source!r} need --gains-file (span-gains.json)")
        gains = {}
        if needs_gains:
            with open(gains_file, encoding="utf-8") as handle:
                gains = json.load(handle)
        rows = {row["id"]: row for row in read_jsonl(train_data)}
        self.items = []
        for span in read_jsonl(spans_file):
            rid = span["id"]
            if rid not in rows:
                raise ValueError(f"span index holds id {rid}, which is not in {train_data}")
            row = rows[rid]
            teacher_prompt = row.get("teacher_prompt", row["prompt"])
            if text_hash(row["prompt"] + row["output"]) != span["hash"] or \
                    text_hash(teacher_prompt + row["output"]) != span["teacher_hash"]:
                raise ValueError(f"id {rid}: text differs from the one indexed (hash mismatch); rebuild the span index")
            if span.get("truncated"):
                raise ValueError(f"id {rid} was truncated when the spans were built; the spans do not cover the sample")
            entry = gains.get(str(rid))
            if needs_gains:
                if entry is None:
                    raise ValueError(f"id {rid} has no entry in {gains_file}")
                if entry["hash"] != span["hash"] or entry["teacher_hash"] != span["teacher_hash"]:
                    raise ValueError(f"id {rid}: {gains_file} was scored on another text than the span index")
            sizes = [b - a for a, b in span["student_tokens"]]
            if sizes != [b - a for a, b in span["teacher_tokens"]]:
                raise ValueError(f"id {rid}: student and teacher spans differ in size (not one tokenizer)")
            sizes[-1] += 1  # the EOS belongs to the last span (see SpanKDCollator)
            self.items.append({
                "id": rid, "full": row["prompt"] + row["output"], "teacher_full": teacher_prompt + row["output"],
                "student_tokens": span["student_tokens"], "teacher_tokens": span["teacher_tokens"],
                "omega_ce": span_omega(ce_source, sizes, entry, **alloc),
                "omega_kd": span_omega(kd_source, sizes, entry, **alloc)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SpanKDCollator:
    """Tokenizes both models' texts and lists, for every supervised token, where the student and the teacher predict it.

    Position p of a model's logits predicts token p + 1, so span [a, b) is predicted at positions a - 1 .. b - 2; the
    last span also holds the position b - 1 that predicts the EOS. Student and teacher spans have the same token count
    because they share a tokenizer, so the n-th token of a span pairs the two models' positions one to one.
    """

    def __init__(self, student_tokenizer, teacher_tokenizer, max_len, teacher_max_len, needs_teacher=True):
        self.student_tokenizer, self.teacher_tokenizer = student_tokenizer, teacher_tokenizer
        self.max_len, self.teacher_max_len, self.needs_teacher = max_len, teacher_max_len, needs_teacher

    def _encode(self, tokenizer, texts, max_len):
        return tokenizer(texts, add_special_tokens=False, truncation=True, padding=True, max_length=max_len,
                         return_tensors="pt", return_token_type_ids=False)

    def __call__(self, items):
        student = self._encode(self.student_tokenizer, [i["full"] for i in items], self.max_len)
        teacher = self._encode(self.teacher_tokenizer, [i["teacher_full"] for i in items], self.teacher_max_len) \
            if self.needs_teacher else None
        eos = self.student_tokenizer.eos_token_id
        columns = {k: [] for k in ("s_b", "s_p", "t_b", "t_p", "labels", "w_ce", "w_kd")}
        for b, item in enumerate(items):
            ids = student["input_ids"][b]
            lengths = [(int(student["attention_mask"][b].sum()), item["student_tokens"][-1][1], "student")]
            if teacher is not None:
                lengths.append((int(teacher["attention_mask"][b].sum()), item["teacher_tokens"][-1][1], "teacher"))
            for length, expected, name in lengths:
                if length != expected:
                    raise ValueError(f"id {item['id']}: {length} {name} tokens, the span index covers {expected} "
                                     f"(text longer than max_len, or tokenized differently from the index)")
            last = len(item["student_tokens"]) - 1
            for n, ((sa, sb), (ta, _)) in enumerate(zip(item["student_tokens"], item["teacher_tokens"])):
                extra = int(n == last and eos is not None)
                count = sb - sa + extra
                for k in range(count):
                    columns["s_b"].append(b)
                    columns["s_p"].append(sa - 1 + k)
                    columns["t_b"].append(b)
                    columns["t_p"].append(ta - 1 + k)
                    columns["labels"].append(int(ids[sa + k]) if k < sb - sa else eos)
                    columns["w_ce"].append(item["omega_ce"][n] / count)
                    columns["w_kd"].append(item["omega_kd"][n] / count)
        tensors = {k: torch.tensor(v, dtype=torch.float32 if k.startswith("w_") else torch.long)
                   for k, v in columns.items()}
        return {"student": dict(student), "teacher": None if teacher is None else dict(teacher),
                "n_samples": len(items), "ids": [i["id"] for i in items], **tensors}


def distill_terms(student_logits, teacher_logits, labels, top_k, temperature):
    """Per-token CE and forward KL(teacher || student) on the teacher's top-k support, renormalised on it."""
    ce = F.cross_entropy(student_logits, labels, reduction="none")
    values, index = teacher_logits.topk(top_k, dim=-1)
    teacher_log = F.log_softmax(values / temperature, dim=-1)
    student_log = F.log_softmax(student_logits.gather(-1, index) / temperature, dim=-1)
    kl = temperature ** 2 * (teacher_log.exp() * (teacher_log - student_log)).sum(-1)
    return ce, kl


def weighted_loss(batch, student_logits, teacher_logits, beta, top_k, temperature):
    """(1 - beta) * CE + beta * KD with the masses of the collator; per-sample sums of mass 1, mean over samples."""
    ce, kl = distill_terms(student_logits, teacher_logits, batch["labels"], top_k, temperature) \
        if teacher_logits is not None else (F.cross_entropy(student_logits, batch["labels"], reduction="none"), None)
    n = batch["n_samples"]
    ce_loss = (batch["w_ce"] * ce).sum() / n
    kd_loss = ce_loss.new_zeros(()) if kl is None else (batch["w_kd"] * kl).sum() / n
    stats = {"ce": ce_loss.item(), "kd": kd_loss.item(), "ce_token": ce.mean().item(),
             "kd_token": 0.0 if kl is None else kl.mean().item()}
    return (1.0 - beta) * ce_loss + beta * kd_loss, stats


def load_models(args, device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"
    teacher_tokenizer = teacher = None
    if args.kd_beta > 0:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.padding_side = "right"
        if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
            raise ValueError("student and teacher must share one tokenizer")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtypes[args.teacher_dtype]
                                                       ).to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=dtypes[args.student_dtype])
    student = get_peft_model(student, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
                                                 lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=list(LORA_TARGETS)))
    for parameter in student.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()  # fp32 adapters, bf16 matmuls under autocast
    return student.to(device).train(), student_tokenizer, teacher_tokenizer, teacher


def move(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    student, student_tokenizer, teacher_tokenizer, teacher = load_models(args, device)
    vocab = len(student_tokenizer)  # logits rows beyond the tokenizer are padding of the embedding matrix

    alloc = dict(tau=args.alloc_tau, clip=args.alloc_clip, lam=args.alloc_lambda, normalization=args.alloc_norm)
    dataset = SpanKDDataset(args.train_data, args.spans_file, args.gains_file, args.ce_weights, args.kd_weights, alloc)
    collate = SpanKDCollator(student_tokenizer, teacher_tokenizer, args.max_len, args.teacher_max_len,
                             needs_teacher=teacher is not None)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))

    accum = max(1, math.ceil(args.batch_size / args.micro_batch_size))
    total_steps = math.ceil(len(loader) / accum) * args.epochs
    optimizer = optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scheduler = cosine_schedule(optimizer, int(total_steps * args.warmup_ratio), total_steps,
                                args.min_learning_rate / args.learning_rate)
    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, "train_stats.jsonl")
    open(stats_path, "w").close()
    window, step = [], 0

    def optimizer_step():
        nonlocal step
        norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        row = {k: sum(r[k] for r in window) / len(window) for k in window[0]}
        row.update(step=step, lr=scheduler.get_last_lr()[0], grad_norm=float(norm))
        with open(stats_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        window.clear()

    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        micro = 0
        for batch in loader:
            batch = {k: (move(v, device) if isinstance(v, dict) else v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items() if v is not None}
            teacher_logits = None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(**batch["teacher"], use_cache=False).logits[
                            batch["t_b"], batch["t_p"]][:, :vocab].float()
                student_logits = student(**batch["student"], use_cache=False).logits[
                    batch["s_b"], batch["s_p"]][:, :vocab].float()
                loss, stats = weighted_loss(batch, student_logits, teacher_logits, args.kd_beta, args.kd_top_k,
                                            args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {stats}")
            (loss / accum).backward()
            window.append({"loss": loss.item(), **stats})
            micro += 1
            if micro % accum == 0:
                optimizer_step()
        if micro % accum:  # flush the last partial accumulation window
            optimizer_step()

    merged = student.merge_and_unload().to(torch.bfloat16 if device.type == "cuda" else torch.float32)
    merged.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-data", required=True, help="SRA jsonl (prepare_sra_data.py build)")
    p.add_argument("--spans-file", required=True, help="word-span index (span_index.py --span-mode word)")
    p.add_argument("--gains-file", help="span-gains.json (token_gain.py signals); needed for student/teacher/diff weights")
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model", default="", help="not loaded when --kd-beta is 0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ce-weights", choices=STRUCTURAL_SOURCES + GAIN_SOURCES, default="token")
    p.add_argument("--kd-weights", choices=STRUCTURAL_SOURCES + GAIN_SOURCES, default="token")
    p.add_argument("--kd-beta", type=float, default=0.5, help="loss = (1 - beta) CE + beta KD; 0 is plain SFT")
    p.add_argument("--kd-top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--alloc-tau", type=float, default=1.0)
    p.add_argument("--alloc-clip", type=float, default=2.0)
    p.add_argument("--alloc-lambda", type=float, default=0.5)
    p.add_argument("--alloc-norm", choices=("per-trace", "none"), default="per-trace")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128, help="effective batch (micro batches are accumulated)")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024, help="student tokens per sample")
    p.add_argument("--teacher-max-len", type=int, default=2560)
    p.add_argument("--student-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--teacher-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=10)
    args = p.parse_args(argv)
    if not 0.0 <= args.kd_beta <= 1.0:
        p.error("--kd-beta must be in [0, 1]")
    if args.kd_beta > 0 and not args.teacher_model:
        p.error("--teacher-model is required unless --kd-beta is 0")
    return args


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
