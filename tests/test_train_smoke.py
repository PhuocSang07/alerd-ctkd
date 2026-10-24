import json
import os
import random
import sys

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from run_distill_llm import build_trainer, main, parse  # noqa: E402
from span_index import build_rows  # noqa: E402
from llm_train import cosine_schedule  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    rng = random.Random(0)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student_tok = train_tokenizer("bpe", corpus, vocab_size=330)
    teacher_tok = train_tokenizer("bpe", corpus, vocab_size=420)
    student_tok.save_pretrained(root / "student")
    teacher_tok.save_pretrained(root / "teacher")

    common = dict(intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=512)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(transformers.Qwen2Config(vocab_size=len(student_tok), hidden_size=32, **common)
                                     ).save_pretrained(root / "student")
    AutoModelForCausalLM.from_config(transformers.LlamaConfig(vocab_size=len(teacher_tok), hidden_size=48, **common)
                                     ).save_pretrained(root / "teacher")

    rows = make_rows(rng, 7)
    data = root / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    dataset = LLMDataset(str(data), student_tok, teacher_tok)
    index = build_rows(dataset, student_tok, teacher_tok, 256)
    spans = root / "spans.jsonl"
    spans.write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
    gain = {str(r["id"]): [rng.uniform(-2, 3) for _ in range(r["n_steps"])] for r in index}
    (root / "gain.json").write_text(json.dumps(gain))
    return root


def cli(root, out, **overrides):
    opts = {
        "--train_data": root / "train.jsonl", "--max_len": 256, "--batch_size": 2, "--grad_accum_steps": 2,
        "--num_train_epochs": 2, "--learning_rate": 1e-3, "--min_learning_rate": 1e-4,
        "--student_device": "cpu", "--teach_device": "cpu", "--student_dtype": "float32",
        "--teacher_dtype": "float32", "--use_lora": "True", "--lora_rank": 4, "--lora_alpha": 4,
        "--teacher_layers_mapping": "2 3", "--student_encoder_layers_finetuned": "2 3",
        "--hidden_loss_weights": "1 1", "--teacher_embedding_dimension": 48, "--output_dir": out,
        "--teacher_model": root / "teacher", "--teacher_tokenizer": root / "teacher",
        "--student_model": root / "student", "--student_tokenizer": root / "student",
        "--hard_label_loss_weight": 0.5, "--seed": 1, "--student_model_type": "qwen",
        "--teacher_model_type": "llama", "--span_weight_source": "alerd", "--spans_file": root / "spans.jsonl",
        "--gain_file": root / "gain.json",
    }
    opts.update({f"--{k}": v for k, v in overrides.items()})
    argv = []
    for key, value in opts.items():
        argv += [key, *str(value).split(" ")]
    return argv


def test_end_to_end_training_saves_an_hf_checkpoint(workspace, tmp_path):
    out = tmp_path / "run"
    main(cli(workspace, out))

    stats = [json.loads(line) for line in (out / "train_stats.jsonl").read_text().splitlines()]
    assert len(stats) == 2 * 2  # 7 examples / batch 2 = 4 micro-batches = 2 optimizer steps per epoch
    for row in stats:
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        assert {"cosine", "geometry", "kl", "ce", "w_p50", "budget"} <= set(row)
        assert row["budget"] == pytest.approx(1.0, abs=1e-5)  # sum of omega per sample (alerd, per-trace)

    model = AutoModelForCausalLM.from_pretrained(out)  # plain HF checkpoint: LoRA merged, projections dropped
    assert not any("lora" in name for name, _ in model.named_parameters())
    assert (out / "kd" / "proj_hidden_layers.pt").exists()
    assert AutoTokenizer.from_pretrained(out).eos_token == "<eos>"


def test_saved_checkpoint_reproduces_the_trained_function(workspace, tmp_path):
    args, extras = parse(cli(workspace, tmp_path / "run", use_lora="True"))
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


def test_gradients_reach_student_and_projections_only(workspace, tmp_path):
    args, extras = parse(cli(workspace, tmp_path / "run"))
    trainer = build_trainer(args, extras)
    student_inputs, teacher_inputs, labels = next(iter(trainer.train_loader))
    assert student_inputs["span_weights_alloc"].shape[0] == 2
    teacher_outputs = trainer.get_teacher_eval(teacher_inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = trainer.compute_loss(student_inputs, labels, teacher_outputs)
    loss.backward()
    lora = [p for n, p in trainer.student.model.model.named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora)
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


@pytest.mark.parametrize("source", ["attention", "uniform"])
def test_other_sources_train_without_score_files(workspace, tmp_path, source):
    out = tmp_path / source
    main(cli(workspace, out, span_weight_source=source, num_train_epochs=1, use_lora="False"))
    assert (out / "model.safetensors").exists()


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
    spans = {r["id"]: r for r in load(workspace / "spans.jsonl")}
    examples = {r["id"]: r for r in load(workspace / "train.jsonl")}
    report, used = build_report(rows, spans, examples)
    assert len(used) == 7 and "Spearman" in report
