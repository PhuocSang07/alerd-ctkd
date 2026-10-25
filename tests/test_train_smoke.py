import json
import os
import random
import sys

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from run_distill_llm import build_trainer, main, parse  # noqa: E402
from span_index import build_rows  # noqa: E402
from llm_train import cosine_schedule  # noqa: E402

COMMON = dict(intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=512)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A Qwen2 student and a GPT-2 student (no pad token) against a Llama teacher, with the offline artefacts of each."""
    root = tmp_path_factory.mktemp("smoke")
    rng = random.Random(0)
    corpus = [make_text(rng, 6) for _ in range(200)]
    teacher_tok = train_tokenizer("bpe", corpus, vocab_size=420)
    tokenizers = {"qwen": train_tokenizer("bpe", corpus, vocab_size=330),
                  "gpt2": train_tokenizer("bpe", corpus, vocab_size=300, pad=False)}
    torch.manual_seed(0)
    teacher_tok.save_pretrained(root / "teacher")
    AutoModelForCausalLM.from_config(transformers.LlamaConfig(vocab_size=len(teacher_tok), hidden_size=48, **COMMON)
                                     ).save_pretrained(root / "teacher")

    rows = make_rows(rng, 7)
    data = root / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (root / "gain-teacher.json").write_text(json.dumps(make_gain_file(rows, teacher_tok, "teacher", rng)))

    workspace = {"root": root, "data": data, "rows": rows, "teacher_gain": root / "gain-teacher.json"}
    for name, tok in tokenizers.items():
        folder = root / name
        tok.save_pretrained(folder)
        if name == "qwen":
            config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, **COMMON)
        else:
            config = transformers.GPT2Config(vocab_size=len(tok), n_embd=32, n_layer=3, n_head=4, n_positions=512)
        AutoModelForCausalLM.from_config(config).save_pretrained(folder)
        index = build_rows(LLMDataset(str(data), tok, teacher_tok), tok, teacher_tok, 256)
        (folder / "spans.jsonl").write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
        (folder / "gain.json").write_text(json.dumps(make_gain_file(rows, tok, "student", rng)))
        predictability = {str(r["id"]): [rng.uniform(-3, -0.1) for _ in range(r["n_steps"])] for r in index}
        (folder / "predictability.json").write_text(json.dumps(predictability))
        workspace[name] = folder
    return workspace


def cli(w, out, student="qwen", **overrides):
    folder = w[student]
    opts = {
        "--train_data": w["data"], "--max_len": 256, "--batch_size": 2, "--grad_accum_steps": 2,
        "--num_train_epochs": 2, "--learning_rate": 1e-3, "--min_learning_rate": 1e-4,
        "--student_device": "cpu", "--teach_device": "cpu", "--student_dtype": "float32",
        "--teacher_dtype": "float32", "--teacher_layers_mapping": "2 3", "--student_encoder_layers_finetuned": "2 3",
        "--hidden_loss_weights": "1 1", "--teacher_embedding_dimension": 48, "--output_dir": out,
        "--teacher_model": w["root"] / "teacher", "--teacher_tokenizer": w["root"] / "teacher",
        "--student_model": folder, "--student_tokenizer": folder,
        "--hard_label_loss_weight": 0.5, "--seed": 1, "--student_model_type": student,
        "--teacher_model_type": "llama", "--span_weight_source": "alerd", "--spans_file": folder / "spans.jsonl",
        "--student_gain_file": folder / "gain.json", "--teacher_gain_file": w["teacher_gain"],
        "--gain_source": "student", "--gain_unit": "span",
    }
    opts.update({f"--{k}": v for k, v in overrides.items()})
    argv = []
    for key, value in opts.items():
        if value is None:
            continue
        argv += [key, *str(value).split(" ")]
    return argv


def read_stats(out):
    return [json.loads(line) for line in (out / "train_stats.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("student", ["qwen", "gpt2"])
def test_end_to_end_full_fine_tuning_saves_an_hf_checkpoint(workspace, tmp_path, student):
    out = tmp_path / "run"
    main(cli(workspace, out, student))

    stats = read_stats(out)
    assert len(stats) == 2 * 2  # 7 examples / batch 2 = 4 micro-batches = 2 optimizer steps per epoch
    for row in stats:
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        assert {"cosine", "geometry", "kl", "ce", "w_p50", "budget"} <= set(row)
        assert row["budget"] == pytest.approx(1.0, abs=1e-5)  # sum of omega per sample (alerd, per-trace)

    model = AutoModelForCausalLM.from_pretrained(out)  # plain HF checkpoint, projections kept apart
    expected = "GPT2LMHeadModel" if student == "gpt2" else "Qwen2ForCausalLM"
    assert type(model).__name__ == expected
    assert (out / "kd" / "proj_hidden_layers.pt").exists()
    assert AutoTokenizer.from_pretrained(out).eos_token == "<eos>"


def test_lora_with_gradient_checkpointing_still_trains_and_merges(workspace, tmp_path):
    out = tmp_path / "run"
    main(cli(workspace, out, use_lora="True", lora_rank=4, lora_alpha=4, gradient_checkpointing="True",
             num_train_epochs=1))
    assert all(torch.isfinite(torch.tensor(v)) for row in read_stats(out) for v in row.values())
    model = AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("gain_source,gain_unit", [("teacher", "span"), ("diff", "span"), ("student", "step"),
                                                   ("diff", "step")])
def test_every_gain_source_and_unit_trains(workspace, tmp_path, gain_source, gain_unit):
    out = tmp_path / "run"
    main(cli(workspace, out, num_train_epochs=1, gain_source=gain_source, gain_unit=gain_unit))
    for row in read_stats(out):
        assert row["budget"] == pytest.approx(1.0, abs=1e-5)
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())


def test_missing_teacher_gain_file_stops_before_training(workspace, tmp_path):
    with pytest.raises(FileNotFoundError, match="teacher gain file"):
        build_trainer(*parse(cli(workspace, tmp_path / "run", gain_source="diff",
                                 teacher_gain_file=workspace["root"] / "absent.json")))
    with pytest.raises(ValueError, match="student gain file"):
        build_trainer(*parse(cli(workspace, tmp_path / "run", gain_source="diff", student_gain_file=None)))


def test_saved_checkpoint_reproduces_the_trained_function(workspace, tmp_path):
    args, extras = parse(cli(workspace, tmp_path / "run", use_lora="True", lora_rank=4, lora_alpha=4))
    trainer = build_trainer(args, extras)
    model = trainer.student.model.model
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.05)  # make the adapter non-trivial
    ids = torch.randint(1, 100, (1, 12))
    model.eval()
    with torch.no_grad():
        before = model(ids).logits
    trainer.student.save(tmp_path / "ckpt")
    after = AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt").eval()(ids).logits
    torch.testing.assert_close(after.float(), before, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("student", ["qwen", "gpt2"])
def test_gradients_reach_the_student_and_projections_only(workspace, tmp_path, student):
    args, extras = parse(cli(workspace, tmp_path / "run", student))
    trainer = build_trainer(args, extras)
    student_inputs, teacher_inputs, labels = next(iter(trainer.train_loader))
    assert student_inputs["span_weights_alloc"].shape[0] == 2
    teacher_outputs = trainer.get_teacher_eval(teacher_inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = trainer.compute_loss(student_inputs, labels, teacher_outputs)
    loss.backward()
    params = [p for p in trainer.student.model.model.parameters() if p.requires_grad]
    # the layers after the last distilled one all feed lm_head; every student weight is trained under autocast
    assert params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.student.proj_hidden_layers)
    assert all(p.grad is None for p in trainer.teacher_model.model.parameters())


def test_eos_is_supervised_after_the_response(workspace, tmp_path):
    args, extras = parse(cli(workspace, tmp_path / "run", span_weight_source="attention"))
    trainer = build_trainer(args, extras)
    student_inputs, _, labels = next(iter(trainer.train_loader))
    eos = trainer.student_tokenizer.eos_token_id
    lengths = student_inputs["attention_mask"].sum(1)
    for i, n in enumerate(lengths.tolist()):
        assert labels[i, n] == eos and (labels[i, n + 1:] == -100).all()
        assert (labels[i, :n] != -100).sum() > 0  # response tokens are supervised, the prompt is not
        assert labels[i, 0] == -100


def test_gpt2_gets_a_pad_token_from_its_eos(workspace, tmp_path):
    args, extras = parse(cli(workspace, tmp_path / "run", "gpt2"))
    trainer = build_trainer(args, extras)
    assert trainer.student_tokenizer.pad_token == trainer.student_tokenizer.eos_token
    student_inputs, _, _ = next(iter(trainer.train_loader))
    assert student_inputs["attention_mask"].sum(1).min() < student_inputs["attention_mask"].shape[1]


@pytest.mark.parametrize("source", ["attention", "uniform"])
def test_other_sources_train_without_score_files(workspace, tmp_path, source):
    out = tmp_path / source
    main(cli(workspace, out, span_weight_source=source, num_train_epochs=1, student_gain_file=None,
             teacher_gain_file=None))
    assert (out / "model.safetensors").exists()


def test_sft_trains_without_loading_the_teacher(workspace, tmp_path):
    out = tmp_path / "sft"
    argv = cli(workspace, out, hard_label_loss_weight=1.0, num_train_epochs=1, teacher_model=None, teacher_tokenizer=None,
               teacher_layers_mapping=None, student_encoder_layers_finetuned=None, hidden_loss_weights=None,
               span_weight_source="attention", spans_file=None, student_gain_file=None, teacher_gain_file=None,
               teacher_embedding_dimension=None)
    args, extras = parse(argv)
    assert args.sft_only and not args.knowledge_distillation
    trainer = build_trainer(args, extras)
    assert trainer.teacher_model is None and trainer.teacher_tokenizer is None
    student_inputs, teacher_inputs, labels = next(iter(trainer.train_loader))
    assert teacher_inputs is None and "pooler_safe_idx" not in student_inputs
    main(argv)
    for row in read_stats(out):
        assert set(row) >= {"ce", "loss"} and "cosine" not in row and row["loss"] == pytest.approx(row["ce"])
    assert AutoModelForCausalLM.from_pretrained(out) is not None and not (out / "kd").exists()


def test_sft_loss_equals_the_cross_entropy_of_the_distillation_run(workspace, tmp_path):
    """Same weights, same batch: the cross-entropy term is the same in the sdpa SFT path and the eager KD path."""
    sft_args, extras = parse(cli(workspace, tmp_path / "a", hard_label_loss_weight=1.0, teacher_model=None,
                                 teacher_tokenizer=None, teacher_layers_mapping=None,
                                 student_encoder_layers_finetuned=None, hidden_loss_weights=None,
                                 span_weight_source="attention", spans_file=None, student_gain_file=None,
                                 teacher_gain_file=None, teacher_embedding_dimension=None))
    kd_args, kd_extras = parse(cli(workspace, tmp_path / "b", span_weight_source="attention"))
    sft, kd = build_trainer(sft_args, extras), build_trainer(kd_args, kd_extras)
    kd.student.model.model.load_state_dict(sft.student.model.model.state_dict())
    kd_inputs, kd_teacher, labels = next(iter(kd.train_loader))
    sft_inputs = {k: v for k, v in kd_inputs.items() if k not in ("pooler_safe_idx", "pooler_mask", "span_weights_alloc")}
    _, ce_sft = sft.compute_loss(sft_inputs, labels)  # sdpa attention, no teacher
    _, ce_kd = kd.compute_loss(kd_inputs, labels, kd.get_teacher_eval(kd_teacher))  # eager attention
    assert ce_sft.item() == pytest.approx(ce_kd.item(), rel=1e-4)


def test_max_len_beyond_the_context_and_missing_layers_are_rejected(workspace, tmp_path):
    with pytest.raises(ValueError, match="exceeds the (teacher|student) context"):
        build_trainer(*parse(cli(workspace, tmp_path / "a", "gpt2", max_len=1024)))
    with pytest.raises(ValueError, match="out of range"):
        build_trainer(*parse(cli(workspace, tmp_path / "b", student_encoder_layers_finetuned="2 4")))
    with pytest.raises(ValueError, match="out of range"):
        build_trainer(*parse(cli(workspace, tmp_path / "c", teacher_layers_mapping="2 9")))


def test_cosine_schedule_warms_up_then_decays_to_the_floor():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = cosine_schedule(opt, warmup_steps=2, total_steps=10, min_ratio=0.2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.0 and lrs[2] == pytest.approx(1.0) and lrs[-1] == pytest.approx(0.2, abs=0.05)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))


def test_dump_feeds_the_weight_analysis(workspace, tmp_path):
    from analyze_weights import build_report

    out = tmp_path / "dump_run"
    out.mkdir()
    dump = out / "span_weights.jsonl"
    main(cli(workspace, out, num_train_epochs=1, dump_span_weights=dump))
    load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]  # noqa: E731
    rows = load(dump)
    assert rows and all(abs(sum(r["attention"]) - 1) < 1e-4 and abs(sum(r["omega"]) - 1) < 1e-4 for r in rows)
    spans = {r["id"]: r for r in load(workspace["qwen"] / "spans.jsonl")}
    examples = {r["id"]: r for r in load(workspace["data"])}
    report, used = build_report(rows, spans, examples)
    assert len(used) == 7 and "Spearman" in report and "Weight by span type" in report
