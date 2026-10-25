# ALERD-CTKD: answer-gain span weights for cross-tokenizer distillation, on top of SRA

This repository started from **SRA** (Span Representation Alignment, Dao et al.), which distills a teacher into a
student with a different tokenizer by aligning *spans* that both tokenizers cut at the same characters. It now also
implements **ALERD-CTKD**: the budget-preserving, answer-gain weights of ALERD (Eq. 3-5) replace SRA's attention weights
(Eq. 8) in the three span-level KD losses (cosine, geometry, KL), with the gain measured **per span**. The
cross-entropy loss is unchanged.

Setting: Qwen2.5-7B-Instruct (teacher, no extra SFT) teaches **GPT2-XL** and **Llama-3.2-1B-Instruct** (full
fine-tuning) on **GSM8K train** (7,473 problems). Evaluation: GSM8K test (1,319) and MATH500, n=3 samples, pass@1 and
pass@3 graded by Math-Verify.

Data preparation, gain scoring and evaluation call the `sgl` package (the SpectralGuidedLearning repository) as a
library; point `SGL_SRC` at its `src` directory. Nothing here modifies it.

## 1. Environment

Python >= 3.9, PyTorch >= 2.4, transformers >= 4.44.

```
pip install -r requirements.txt
```

GPU steps (gain scoring and evaluation through `sgl`) also need vLLM. The Hugging Face token is read from the
`HF_TOKEN` environment variable (Llama-3.2 is gated); it is never passed on a command line or written to a script.

## 2. What changed from the original SRA code

| Area | Now |
|---|---|
| Attention | As in SRA: `output_attentions=True` with eager attention for the teacher and the student. Only the weights of the distilled layers are used. |
| Last token | The row of the last *real* token is taken from `attention_mask`, not `[:, -1]` (a pad token for every sequence shorter than the batch with right padding). |
| Hidden state of the last layer | HF's last `hidden_states` entry, i.e. after the final norm, for teacher and student alike. |
| Precision | fp32 master weights plus bf16 autocast; no GradScaler. |
| KL | Computed on the `lm_head` rows of the shared vocabulary only; the teacher never returns full-sequence logits. |
| CE | Prompt masked, EOS supervised after the response. |
| Student | GPT-2 is supported again (no pad token, no chat template) next to Llama/Qwen-style decoders. |
| Teacher prompt | The teacher sees the question in its own chat template; spans are aligned on the shared output text. |
| SFT | `hard_label_loss_weight 1.0` skips the teacher entirely (not loaded, not run). |
| Checkpoint | Plain HF directory (bf16). The KD projections go to `kd/`. |
| Removed | The Dolly evaluator in the training loop; LoRA and gradient checkpointing are still available (`--use_lora`, `--gradient_checkpointing`) but off by default. |

## 3. Data and prompt formats

`prepare_sra_data.py clean` turns each GSM8K row into: the solution without the calculator annotations `<<...>>`, the
`#### N` line replaced by `The final answer is \boxed{N}.` (thousands separators dropped from `N`), `answer` = `N`.
Every line is a step.

The prompt is the instruction `Please reason step by step, and put your final answer within \boxed{}.` followed
directly by the question (the prompt of `sgl.eval`):

* models **with a chat template** (Llama-3.2-Instruct, Qwen2.5-Instruct): the instruction is one user turn rendered with
  `add_generation_prompt=True`; the Llama date line is fixed so that train and eval prompts match on any day;
* **GPT-2** (no chat template, no pad token, no BOS): the **raw** instruction + question, and the solution follows
  immediately. The pad token is set to EOS for batching and EOS is supervised after the response. `max_len` must stay
  <= 1,024 (checked at start-up); evaluation generates at most 512 tokens (`MAX_TOKENS`).

Evaluation uses the same prompts (`run_sgl_eval.py`), so the chat-template flags of `sgl.eval` are not needed. GPT-2 will
score very low; its role is the comparison between methods.

Prompt and output are tokenized jointly (as the trainer does); the build step fails if a token would straddle the
prompt/output boundary.

## 4. Pipeline

Steps marked **GPU** need a GPU (the rest run on CPU). Run them in this order.

1. **Data and gain scoring** (GPU, vLLM):
   ```
   SGL_SRC=/path/to/SpectralGuidedLearning/src bash scripts/score_gain.sh
   ```
   Output: `work/gsm8k-clean.jsonl`; per student `work/<student>/{train-sra.jsonl, tokens-<student>.jsonl,
   gain-tokens-<student>.json, predictability.json}`; once for the teacher `work/Qwen2.5-7B-Instruct/gain-tokens-Qwen2.5-7B-Instruct.json`.
   The gain is measured with `sgl.signals.answer_gain` at **every token boundary** of the scoring model (each response
   token is one "step", `t = 0` included) with the probe `"\n\nTherefore, the final answer is \boxed{"` and the target
   `N}`. A span's gain is `b_end - b_start` of its token range in the scoring model's own tokenization, a step's gain the
   sum over its tokens, so both come from the same `b_t`. Each file records the sha1 of the text it was scored on and the
   tokenizer name, and the trainer stops if they do not match the span index.
2. **Train** (GPU). Defaults: teacher layer 28 of Qwen2.5-7B (`TEACHER_LAYER`, `TEACHER_DIM=3584`), last layer of the
   student (48 for GPT2-XL, 16 for Llama-3.2-1B), 3 epochs, lr 5e-5 cosine to 1e-5, warm-up 10%, effective batch
   32 (8 x 4), `max_len` 512:
   ```
   GAIN_SOURCE=student bash scripts/alerd_ctkd_train.sh llama-3.2-1b alerd-ctkd [seed]
   ```
   Methods: `sft`, `sra`, `sra-uniform`, `alerd-ctkd` (main table) and the ablations listed at the top of
   `scripts/alerd_ctkd_train.sh`. The first call also builds the span index (`span_index.py`, CPU), which pairs every span
   with its step and checks that no sample loses its `\boxed{}`. `GAIN_SOURCE` is `student`, `teacher` or `diff`
   (teacher minus student on raw span gains); `GAIN_UNIT` is `span` (default) or `step` (spans inherit the gain of their
   step). A missing gain file stops the run with the name of the file. `DUMP_SPAN_WEIGHTS=1` also writes the per-span
   attention and omega used by `analyze_weights.py`. Output: `outputs/<student>/<method>/seed-<seed>/`.
3. **Evaluate** (GPU, vLLM). Zero-shot, teacher and trained checkpoints under the ALERD protocol:
   ```
   SGL_SRC=... bash scripts/alerd_ctkd_eval.sh outputs/llama-3.2-1b/alerd-ctkd/seed-42 llama-3.2-1b-alerd-ctkd-s42 meta-llama/Llama-3.2-1B-Instruct
   ```
   `scripts/alerd_ctkd_run_all.sh <student> main|ablations` chains zero-shot, teacher, training and evaluation (three seeds
   for the main table, one seed for ablations).
4. **Tables and analysis** (CPU):
   ```
   python collect_results.py --results-dir results --table main --output tables/main.tex
   python collect_results.py --results-dir results --table ablations --output tables/ablations.tex
   python analyze_gain.py --spans work/Llama-3.2-1B-Instruct/spans-Qwen2.5-7B-Instruct.jsonl \
       --data work/Llama-3.2-1B-Instruct/train-sra.jsonl \
       --student-gain work/Llama-3.2-1B-Instruct/gain-tokens-Llama-3.2-1B-Instruct.json \
       --teacher-gain work/Qwen2.5-7B-Instruct/gain-tokens-Qwen2.5-7B-Instruct.json --output plans/reports/analysis-gain.md
   python analyze_weights.py --dump outputs/llama-3.2-1b/alerd-ctkd/seed-42/span_weights.jsonl \
       --spans work/Llama-3.2-1B-Instruct/spans-Qwen2.5-7B-Instruct.jsonl \
       --data work/Llama-3.2-1B-Instruct/train-sra.jsonl --output plans/reports/analysis-weights.md --html example.html
   ```

Span weights (`--span_weight_source`): `attention` (SRA), `uniform`, `alerd`, `random`, `predictability-easy`,
`predictability-hard`; `--gain_source`, `--gain_unit`, `--alloc_tau`, `--alloc_clip`, `--alloc_lambda`, `--alloc_norm`
(`per-trace`, `corpus`, `none`) and `--answer_only_unit` control the allocation (defaults: c=2, tau=1, lambda=0.5,
per-trace). Every span has mass 1 in the span unit, so `omega_n = w_n / N` and `sum_n omega_n = 1`; with the step unit a
step weighs as many spans as it holds. With `--alloc_lambda 0` the three KD losses equal `uniform`. Predictability scores
are per solution line whatever the gain unit.

The collator refuses to run if a sample's text hash (student or teacher) or span count differs from `spans.jsonl`, so
rebuild the span index whenever the data, the tokenizers or `--max_len` change.

## 5. Tests

```
python -m pytest tests
```

They run offline on CPU with small randomly initialised models (Qwen2, Llama, GPT-2) and tokenizers trained on the spot;
set `SGL_SRC` to also compare the prompt text with `sgl`'s. They cover the attention row of the last real token under
padding, the KD losses against the original SRA code, the ALERD toy table, the GSM8K conversion, the span index for a
GPT-2-style and a Llama-style tokenizer, the gain arithmetic (span gains add up to step gains, `diff` is a difference,
missing or mismatched files stop the run), a full training run for each student type and for SFT, the shell scripts
(with `echo` standing in for Python) and the table and analysis scripts. GPU memory, speed and the quality of the trained
models are not covered.

## 6. Original SRA data format

Training data is `.jsonl`; this fork reads `prompt`, `output` and, for ALERD-CTKD, `id`, `teacher_prompt` (default:
`prompt`) and `step_char_ends` (written by `prepare_sra_data.py build`).
