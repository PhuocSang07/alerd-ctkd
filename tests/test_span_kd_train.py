import json
import os
import random
import sys

import pytest
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from span_index import build_rows  # noqa: E402
from span_kd_train import (SpanKDCollator, SpanKDDataset, distill_terms, main, span_omega, weighted_loss)  # noqa: E402
from token_gain import main as token_gain_main  # noqa: E402

COMMON = dict(intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=512)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A tiny Qwen2 student and teacher on one tokenizer, with the word-span index and the span gains of 10 samples."""
    root = tmp_path_factory.mktemp("span-kd")
    rng = random.Random(0)
    tok = train_tokenizer("bpe", [make_text(rng, 6) for _ in range(200)], vocab_size=330)
    torch.manual_seed(0)
    for name, hidden in (("student", 32), ("teacher", 48)):
        tok.save_pretrained(root / name)
        AutoModelForCausalLM.from_config(transformers.Qwen2Config(vocab_size=len(tok), hidden_size=hidden, **COMMON)
                                         ).save_pretrained(root / name)
    rows = make_rows(rng, 10)  # the teacher's prompt is longer than the student's
    data = root / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index = build_rows(LLMDataset(str(data), tok, tok), tok, tok, 256, span_mode="word")
    (root / "spans.jsonl").write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
    for side in ("student", "teacher"):
        (root / f"gain-{side}.json").write_text(json.dumps(make_gain_file(rows, tok, side, rng)))
    token_gain_main(["signals", "--spans", str(root / "spans.jsonl"), "--student-gains", str(root / "gain-student.json"),
                     "--teacher-gains", str(root / "gain-teacher.json"), "--output", str(root / "span-gains.json")])
    return {"root": root, "tok": tok, "rows": rows, "index": index, "data": data}


def dataset(w, ce="student", kd="diff", **alloc):
    root = w["root"]
    return SpanKDDataset(str(w["data"]), str(root / "spans.jsonl"), str(root / "span-gains.json"), ce, kd, alloc)


def collate(w, items, needs_teacher=True):
    tok = w["tok"]
    tok.padding_side = "right"
    return SpanKDCollator(tok, tok, 256, 256, needs_teacher)(items)


def cli(w, out, **overrides):
    root = w["root"]
    opts = {"train-data": w["data"], "spans-file": root / "spans.jsonl", "gains-file": root / "span-gains.json",
            "student-model": root / "student", "teacher-model": root / "teacher", "output-dir": out,
            "ce-weights": "student", "kd-weights": "diff", "kd-top-k": 20, "epochs": 2, "batch-size": 4,
            "micro-batch-size": 4, "learning-rate": 1e-3, "min-learning-rate": 1e-4, "max-len": 256,
            "teacher-max-len": 256, "student-dtype": "float32", "teacher-dtype": "float32", "device": "cpu", "seed": 1}
    opts.update({k.replace("_", "-"): v for k, v in overrides.items()})
    return [arg for k, v in opts.items() if v is not None for arg in (f"--{k}", str(v))]


def read_stats(out):
    return [json.loads(line) for line in (out / "train_stats.jsonl").read_text().splitlines()]


# ---- the mass of each span -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["token", "uniform", "student", "teacher", "diff"])
def test_every_weighting_gives_each_sample_a_total_mass_of_one(source):
    sizes = [1, 3, 1, 2, 1, 1]
    gains = {"student": [0.5, -1.0, 2.0, 0.1, 0.0, 1.0], "teacher": [0.1] * 3 + [1.0] * 3,
             "diff": [-0.4, 0.7, 0.2, -2.0, 1.5, 0.0]}
    omega = span_omega(source, sizes, gains)
    assert len(omega) == len(sizes) and sum(omega) == pytest.approx(1.0) and min(omega) > 0


def test_token_mass_follows_span_size_and_uniform_does_not():
    assert span_omega("token", [1, 3], None) == pytest.approx([0.25, 0.75])
    assert span_omega("uniform", [1, 3], None) == pytest.approx([0.5, 0.5])


def test_a_larger_gain_gets_a_larger_mass_and_lambda_zero_is_uniform():
    gains = {"student": [-1.0, 0.0, 3.0, 0.5]}
    omega = span_omega("student", [1] * 4, gains)
    assert omega.index(max(omega)) == 2 and omega.index(min(omega)) == 0
    assert span_omega("student", [1] * 4, gains, lam=0.0) == pytest.approx([0.25] * 4)


def test_a_gain_weighting_without_gains_or_with_the_wrong_length_is_rejected(workspace):
    root = workspace["root"]
    with pytest.raises(ValueError, match="gains-file"):
        SpanKDDataset(str(workspace["data"]), str(root / "spans.jsonl"), None, "student", "diff")
    with pytest.raises(ValueError, match="gains for"):
        span_omega("student", [1, 1, 1], {"student": [0.0, 1.0]})


# ---- the dataset and the collator ----------------------------------------------------------------------------------

def test_dataset_masses_come_from_the_gains_of_the_right_signal(workspace):
    signals = json.loads((workspace["root"] / "span-gains.json").read_text())
    for item in dataset(workspace):
        gains = signals[str(item["id"])]
        n = len(gains["student"])
        assert item["omega_ce"] == pytest.approx(span_omega("student", [1] * n, gains))
        assert item["omega_kd"] == pytest.approx(span_omega("diff", [1] * n, gains))


def test_dataset_stops_when_the_text_differs_from_the_span_index(workspace, tmp_path):
    rows = [dict(r) for r in workspace["rows"]]
    rows[0]["output"] += " Extra."
    data = tmp_path / "other.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    root = workspace["root"]
    with pytest.raises(ValueError, match="hash mismatch"):
        SpanKDDataset(str(data), str(root / "spans.jsonl"), str(root / "span-gains.json"), "student", "diff")


def test_dataset_stops_when_the_gains_were_scored_on_another_text(workspace, tmp_path):
    signals = json.loads((workspace["root"] / "span-gains.json").read_text())
    signals["0"]["hash"] = "0" * 40
    path = tmp_path / "gains.json"
    path.write_text(json.dumps(signals))
    root = workspace["root"]
    with pytest.raises(ValueError, match="another text"):
        SpanKDDataset(str(workspace["data"]), str(root / "spans.jsonl"), str(path), "student", "diff")


def test_each_supervised_token_pairs_the_two_models_positions_of_the_same_token(workspace):
    data = dataset(workspace)
    batch = collate(workspace, [data[i] for i in range(6)])
    student_ids, teacher_ids = batch["student"]["input_ids"], batch["teacher"]["input_ids"]
    eos = workspace["tok"].eos_token_id
    assert not torch.equal(batch["s_p"], batch["t_p"])  # the teacher's prompt is longer
    for b, item in enumerate(data[i] for i in range(6)):
        mask = batch["s_b"] == b
        s_p, t_p, labels = batch["s_p"][mask], batch["t_p"][mask], batch["labels"][mask]
        response = item["student_tokens"][-1][1] - item["student_tokens"][0][0]
        assert len(labels) == response + 1  # every response token and the EOS
        assert torch.equal(labels[:-1], student_ids[b, s_p[:-1] + 1]) and labels[-1] == eos
        assert torch.equal(labels[:-1], teacher_ids[b, t_p[:-1] + 1])  # the teacher predicts the same tokens
        assert s_p[-1] == item["student_tokens"][-1][1] - 1 and t_p[-1] == item["teacher_tokens"][-1][1] - 1
        assert batch["w_ce"][mask].sum().item() == pytest.approx(1.0)
        assert batch["w_kd"][mask].sum().item() == pytest.approx(1.0)


def test_a_span_shares_its_mass_equally_between_its_tokens(workspace):
    data = dataset(workspace, "student", "diff")
    item = data[0]
    batch = collate(workspace, [item])
    (a, c) = item["student_tokens"][1]
    mask = (batch["s_p"] >= a - 1) & (batch["s_p"] < c - 1)
    assert mask.sum() == c - a
    assert batch["w_ce"][mask].sum().item() == pytest.approx(item["omega_ce"][1])
    assert batch["w_ce"][mask].unique().numel() == 1


def test_a_text_longer_than_max_len_stops_the_collator(workspace):
    tok = workspace["tok"]
    with pytest.raises(ValueError, match="longer than max_len"):
        SpanKDCollator(tok, tok, 8, 256)([dataset(workspace)[0]])


# ---- the losses ----------------------------------------------------------------------------------------------------

def test_distill_terms_match_a_hand_computed_kl_on_the_teacher_top_k():
    teacher = torch.tensor([[3.0, 1.0, 0.0, -2.0]])
    student = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    temperature = 2.0
    ce, kl = distill_terms(student, teacher, torch.tensor([1]), top_k=2, temperature=temperature)
    q = F.softmax(torch.tensor([3.0, 1.0]) / temperature, -1)  # teacher top-2: ids 0, 1
    p = F.softmax(torch.tensor([0.0, 1.0]) / temperature, -1)
    assert kl.item() == pytest.approx(temperature ** 2 * (q * (q / p).log()).sum().item())
    assert ce.item() == pytest.approx(F.cross_entropy(student, torch.tensor([1])).item())


def test_the_kl_is_zero_for_equal_logits_and_the_full_kl_when_k_is_the_vocabulary():
    logits = torch.randn(5, 11)
    _, kl = distill_terms(logits, logits, torch.zeros(5, dtype=torch.long), top_k=4, temperature=1.0)
    assert kl.abs().max().item() < 1e-6
    other = torch.randn(5, 11)
    _, kl = distill_terms(other, logits, torch.zeros(5, dtype=torch.long), top_k=11, temperature=1.0)
    full = F.kl_div(F.log_softmax(other, -1), F.softmax(logits, -1), reduction="none").sum(-1)
    assert kl == pytest.approx(full.tolist(), abs=1e-5)


def test_the_loss_mixes_ce_and_kd_with_beta_and_beta_zero_is_the_span_weighted_ce(workspace):
    batch = collate(workspace, [dataset(workspace)[i] for i in range(4)])
    vocab = len(workspace["tok"])
    n = len(batch["labels"])
    student, teacher = torch.randn(n, vocab), torch.randn(n, vocab)
    ce_only, _ = weighted_loss(batch, student, None, 0.0, 10, 1.0)
    ce, kl = distill_terms(student, teacher, batch["labels"], 10, 1.0)
    expected_ce = (batch["w_ce"] * ce).sum() / 4
    expected_kd = (batch["w_kd"] * kl).sum() / 4
    assert ce_only.item() == pytest.approx(expected_ce.item())
    mixed, stats = weighted_loss(batch, student, teacher, 0.3, 10, 1.0)
    assert mixed.item() == pytest.approx((0.7 * expected_ce + 0.3 * expected_kd).item())
    assert stats["ce"] == pytest.approx(expected_ce.item()) and stats["kd"] == pytest.approx(expected_kd.item())


def test_token_weights_make_the_ce_the_mean_over_each_samples_tokens(workspace):
    data = dataset(workspace, "token", "token")
    batch = collate(workspace, [data[i] for i in range(4)])
    student = torch.randn(len(batch["labels"]), len(workspace["tok"]))
    loss, _ = weighted_loss(batch, student, None, 0.0, 10, 1.0)
    ce = F.cross_entropy(student, batch["labels"], reduction="none")
    per_sample = [ce[batch["s_b"] == b].mean() for b in range(4)]
    assert loss.item() == pytest.approx(torch.stack(per_sample).mean().item())


# ---- training ------------------------------------------------------------------------------------------------------

def test_end_to_end_training_saves_a_merged_hf_checkpoint(workspace, tmp_path):
    out = tmp_path / "run"
    main(cli(workspace, out))
    stats = read_stats(out)
    assert len(stats) == 6  # 10 samples, micro batches of 4 -> 3 steps per epoch, 2 epochs
    assert all(torch.isfinite(torch.tensor(r["loss"])) and r["kd"] > 0 and r["ce"] > 0 for r in stats)
    assert stats[0]["lr"] > 0 and (out / "run_config.json").exists()
    trained = AutoModelForCausalLM.from_pretrained(out)
    base = AutoModelForCausalLM.from_pretrained(workspace["root"] / "student")
    assert not hasattr(trained, "peft_config")  # LoRA is merged into plain weights
    changed = [n for (n, a), (_, b) in zip(trained.state_dict().items(), base.state_dict().items()) if not torch.equal(a, b)]
    assert changed and all("proj" in n for n in changed)  # only the LoRA targets moved


def test_training_lowers_the_loss_on_a_fixed_batch(workspace, tmp_path):
    out = tmp_path / "overfit"
    main(cli(workspace, out, epochs=60, **{"batch-size": 10, "micro-batch-size": 10, "lora-rank": 8, "lora-alpha": 64,
                                           "learning-rate": 1e-2, "min-learning-rate": 1e-2, "warmup-ratio": 0.0}))
    stats = read_stats(out)
    assert stats[-1]["ce_token"] < stats[0]["ce_token"] - 0.2 and stats[-1]["loss"] < stats[0]["loss"]


def test_the_weights_change_the_loss_but_not_the_token_means_at_the_first_step(workspace, tmp_path):
    first = {}
    for name, ce, kd in (("gain", "student", "diff"), ("token", "token", "token")):
        out = tmp_path / name
        main(cli(workspace, out, epochs=1, **{"ce-weights": ce, "kd-weights": kd, "batch-size": 4,
                                              "micro-batch-size": 4}))
        first[name] = read_stats(out)[0]  # LoRA starts as the identity, so step 1 sees the same models and batch
    assert first["gain"]["ce_token"] == pytest.approx(first["token"]["ce_token"])
    assert first["gain"]["kd_token"] == pytest.approx(first["token"]["kd_token"])
    assert abs(first["gain"]["ce"] - first["token"]["ce"]) > 1e-5  # the near-uniform tiny model leaves little to reweigh
    assert abs(first["gain"]["kd"] - first["token"]["kd"]) > 1e-7


def test_beta_zero_trains_without_loading_a_teacher(workspace, tmp_path):
    out = tmp_path / "sft"
    main(cli(workspace, out, **{"kd-beta": 0.0, "teacher-model": None, "kd-weights": "token"}))
    stats = read_stats(out)
    assert all(r["kd"] == 0.0 and r["loss"] == pytest.approx(r["ce"]) for r in stats)


def test_a_teacher_is_required_unless_beta_is_zero(workspace, tmp_path):
    with pytest.raises(SystemExit):
        main(cli(workspace, tmp_path / "x", **{"teacher-model": None}))


def test_the_span_dump_has_one_row_per_span_and_sample_and_epoch(workspace, tmp_path):
    out, dump = tmp_path / "run", tmp_path / "spans.jsonl"
    main(cli(workspace, out, **{"dump-spans": dump}))
    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    index = {r["id"]: r for r in workspace["index"]}
    assert {r["epoch"] for r in rows} == {0, 1}
    for epoch in (0, 1):
        for rid, span in index.items():
            mine = [r for r in rows if r["epoch"] == epoch and r["id"] == rid]
            assert [r["span"] for r in mine] == list(range(len(span["student_tokens"])))
            assert sum(r["n"] for r in mine) == span["student_tokens"][-1][1] - span["student_tokens"][0][0] + 1
            assert sum(r["omega_ce"] for r in mine) == pytest.approx(1.0)
            assert sum(r["omega_kd"] for r in mine) == pytest.approx(1.0)
            assert all(r["ce"] > 0 and r["kd"] >= 0 for r in mine)
    # the span means rebuild the CE the trainer logged: sum_n omega_n * ce_n averaged over the samples of a step
    first = [r for r in rows if r["epoch"] == 0 and r["step"] == 1]
    ids = {r["id"] for r in first}
    ce = sum(r["omega_ce"] * r["ce"] for r in first) / len(ids)
    assert ce == pytest.approx(read_stats(out)[0]["ce"], rel=1e-4)
