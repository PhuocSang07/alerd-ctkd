# ALERD-CTKD: answer-gain span weights for cross-tokenizer distillation, on top of SRA

This repository started from **SRA** (Span Representation Alignment, Dao et al.), which distills a teacher into a
student with a different tokenizer by aligning *spans* that both tokenizers cut at the same characters. It now also
implements **ALERD-CTKD**: the budget-preserving, answer-gain step weights of ALERD (Eq. 3-5) replace SRA's
attention weights (Eq. 8) in the three span-level KD losses (cosine, geometry, KL). The cross-entropy loss is
unchanged. Training data is s1K-1.1 with the `deepseek_attempt` solutions; evaluation follows ALERD (AIME25, AIME24,
AMC12, MATH500, pass@1 and pass@3).

Data preparation, gain scoring and evaluation call the `sgl` package (the SpectralGuidedLearning repository) as a
library; point `SGL_SRC` at its `src` directory. Nothing here modifies it.

## 1. Environment

Python >= 3.9, PyTorch >= 2.4, a transformers version that knows your student (Qwen3 needs >= 4.51).

```
pip install -r requirements.txt
```

GPU steps also need vLLM (gain scoring and evaluation, through `sgl`). The Hugging Face token is read from the
`HF_TOKEN` environment variable; it is never passed on a command line or written to a script.

## 2. What changed from the original SRA code

| Area | Now |
|---|---|
| Attention | Fused SDPA in every layer. The distilled layers carry hooks (`layer_capture.py`) that store the layer output and recompute the attention row of the last *real* token (RoPE, GQA, padding), so no T x T matrix is built. Used for teacher and student, for SRA and ALERD-CTKD alike. |
| Last token | The index comes from `attention_mask`, not `[:, -1]`, which is a pad token with right padding. |
| Hidden state of the last layer | Goes through the final norm for teacher and student, as in HF `output_hidden_states`. |
| Precision | fp32 master weights plus bf16 autocast; no GradScaler. LoRA students keep a bf16 base. Gradient checkpointing, micro-batch 1 with gradient accumulation to 32. |
| KL | Computed on the `lm_head` rows of the shared vocabulary only; the teacher never returns full-sequence logits. |
| CE | Prompt masked, EOS supervised after the response. |
| Length | `max_len` defaults to 2500 and prompts are no longer cut to `max_len // 2`; `span_index.py` fails if a sample would lose its final `\boxed{}`. |
| Checkpoint | Plain HF directory (bf16, LoRA merged). The KD projections go to `kd/`. |
| Removed | Student/teacher architectures without a Llama-style decoder (GPT-2, OPT) and the Dolly evaluator in the training loop. |

## 3. Pipeline

Steps marked **GPU** need a GPU (the rest run on CPU). Run them in this order.

1. **Data, gain and predictability scores, per student** (GPU). Filters s1K-1.1 to the rows whose
   `deepseek_attempt` has an extractable final `\boxed{}`, segments them into steps with the student's chat
   template (thinking off, P-ALIGN prompt), scores answer gain and per-step log-probability with the frozen student,
   and writes the SRA jsonl:
   ```
   SGL_SRC=/path/to/SpectralGuidedLearning/src bash scripts/score_gain.sh
   ```
   Output: `work/<student>/{train-segmented.jsonl,gain.json,predictability.json,train-sra.jsonl}`.
2. **Train** (GPU). The teacher is yours to choose: any Llama-style decoder with a tokenizer different from Qwen's.
   ```
   TEACHER=<hf id> TEACHER_LAYER=<1-indexed layer> TEACHER_DIM=<hidden size> \
   bash scripts/alerd_ctkd_train.sh r1-qwen-1.5b alerd-ctkd
   ```
   The first call also builds the span index (`span_index.py`, CPU), which pairs every span with its step and
   checks that no sample loses its `\boxed{}`. Methods: `sra`, `sra-uniform`, `alerd-ctkd` (main table) and the
   ablations listed at the top of `scripts/alerd_ctkd_train.sh`. Set `DUMP_SPAN_WEIGHTS=1` to also write the
   per-span attention and omega used by `analyze_weights.py`.
3. **Evaluate** (GPU, vLLM). Zero-shot, teacher and trained checkpoints under the ALERD protocol:
   ```
   SGL_SRC=... bash scripts/alerd_ctkd_eval.sh outputs/r1-qwen-1.5b/alerd-ctkd r1-qwen-1.5b-alerd-ctkd deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```
   `scripts/alerd_ctkd_run_all.sh <student> main|ablations` chains zero-shot, teacher, training and evaluation.
4. **Tables and analysis** (CPU):
   ```
   python collect_results.py --results-dir results --table main --output tables/main.tex
   python collect_results.py --results-dir results --table ablations --output tables/ablations.tex
   python analyze_weights.py --dump outputs/r1-qwen-1.5b/alerd-ctkd/span_weights.jsonl \
       --spans work/DeepSeek-R1-Distill-Qwen-1.5B/spans-<teacher>.jsonl \
       --data work/DeepSeek-R1-Distill-Qwen-1.5B/train-sra.jsonl --output plans/reports/analysis.md --html example.html
   ```

Span weights (`--span_weight_source`): `attention` (SRA), `uniform`, `alerd`, `random`, `predictability-easy`,
`predictability-hard`; `--alloc_tau`, `--alloc_clip`, `--alloc_lambda`, `--alloc_norm` (`per-trace`, `corpus`,
`none`) and `--answer_only_unit` control the allocation (defaults: c=2, tau=1, lambda=0.5, per-trace). With
`--span_weight_source alerd` and `--alloc_lambda 0` the three KD losses equal `uniform`.

The collator refuses to run if a sample's text hash or span count differs from `spans.jsonl`, so rebuild the span
index whenever the data, the tokenizers or `--max_len` change.

## 4. Tests

```
python -m pytest tests
```

They run offline on CPU with small randomly initialised models and tokenizers trained on the spot. They cover the
last-row attention against eager attention, the KD losses against the original SRA code, the ALERD toy table, the
span index, a full training run with LoRA and gradient checkpointing, and the table and analysis scripts. GPU memory,
speed and the quality of the trained models are not covered.

## 5. Original SRA data format

Training data is `.jsonl`; this fork reads `prompt`, `output` and, for ALERD-CTKD, `id` and `step_char_ends`
(written by `prepare_sra_data.py convert`).
