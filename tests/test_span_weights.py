import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from allocation import trace_weights  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from span_index import build_rows  # noqa: E402
from span_weights import SpanWeightProvider  # noqa: E402
from token_gain import GainFile, span_gains  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = random.Random(5)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student, teacher = train_tokenizer("bpe", corpus), train_tokenizer("bpe", corpus, vocab_size=420)
    rows = make_rows(rng, 6)
    root = tmp_path_factory.mktemp("provider")
    data = root / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index = build_rows(LLMDataset(str(data), student, teacher), student, teacher, 256)
    (root / "spans.jsonl").write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
    for side, tokenizer in (("student", student), ("teacher", teacher)):
        (root / f"gain-{side}.json").write_text(json.dumps(make_gain_file(rows, tokenizer, side, rng, model=side)))
    predictability = {str(r["id"]): [rng.uniform(-3, -0.1) for _ in range(r["n_steps"])] for r in index}
    (root / "predictability.json").write_text(json.dumps(predictability))
    return {"root": root, "index": index, "rows": rows, "data": str(data)}


def provider(world, source="alerd", gain_source="student", gain_unit="span", **kw):
    root = world["root"]
    return SpanWeightProvider(source, str(root / "spans.jsonl"), world["data"],
                              student_gain_file=str(root / "gain-student.json"),
                              teacher_gain_file=str(root / "gain-teacher.json"),
                              predictability_file=str(root / "predictability.json"),
                              gain_source=gain_source, gain_unit=gain_unit, **kw)


def raw_span_gains(world, row, role):
    gains = GainFile(str(world["root"] / f"gain-{role}.json"), role)
    entry = gains.entry(row["id"], row["hash"] if role == "student" else row["teacher_hash"],
                        row[f"{role}_tokens"][0][0], row[f"{role}_tokens"][-1][1])
    return span_gains(entry, row[f"{role}_tokens"])


GAIN_CONFIGS = [(s, u) for s in ("student", "teacher", "diff") for u in ("span", "step")]


@pytest.mark.parametrize("gain_source,gain_unit", GAIN_CONFIGS)
def test_weights_sum_to_one_per_sample(world, gain_source, gain_unit):
    built = provider(world, gain_source=gain_source, gain_unit=gain_unit)
    for row in world["index"]:
        omega = built.omega[row["id"]]
        assert len(omega) == len(row["student_tokens"]) and sum(omega) == pytest.approx(1.0)
        assert min(omega) > 0


@pytest.mark.parametrize("gain_source,gain_unit", GAIN_CONFIGS)
def test_lambda_zero_is_uniform_for_every_gain_source_and_unit(world, gain_source, gain_unit):
    built = provider(world, gain_source=gain_source, gain_unit=gain_unit, lam=0.0)
    uniform = provider(world, "uniform")
    for row in world["index"]:
        assert built.omega[row["id"]] == pytest.approx(uniform.omega[row["id"]])
        n = len(row["student_tokens"])
        assert uniform.omega[row["id"]] == pytest.approx([1.0 / n] * n)


def test_span_unit_weights_follow_alerd_on_the_span_gains(world):
    built = provider(world, gain_source="student", tau=2.0, lam=0.3)
    for row in world["index"]:
        gains = raw_span_gains(world, row, "student")
        n = len(gains)
        w = trace_weights(gains, [1] * n, tau=2.0, lam=0.3)
        assert built.omega[row["id"]] == pytest.approx([x / n for x in w])


def test_diff_is_the_teacher_minus_the_student_on_raw_span_gains(world):
    built = provider(world, gain_source="diff")
    for row in world["index"]:
        diff = [t - s for t, s in zip(raw_span_gains(world, row, "teacher"), raw_span_gains(world, row, "student"))]
        n = len(diff)
        assert built.omega[row["id"]] == pytest.approx([x / n for x in trace_weights(diff, [1] * n)])
        assert built.omega[row["id"]] != pytest.approx(provider(world, gain_source="student").omega[row["id"]])


def test_teacher_source_uses_the_teacher_side_tokens(world):
    built = provider(world, gain_source="teacher")
    for row in world["index"]:
        gains = raw_span_gains(world, row, "teacher")
        n = len(gains)
        assert built.omega[row["id"]] == pytest.approx([x / n for x in trace_weights(gains, [1] * n)])


def test_step_unit_spans_inherit_their_steps_weight_and_budget_counts_spans(world):
    built = provider(world, gain_unit="step")
    entries = json.loads((world["root"] / "gain-student.json").read_text())["samples"]
    for row in world["index"]:
        masses = [row["step_of_span"].count(k) for k in range(row["n_steps"])]
        w = trace_weights(entries[str(row["id"])]["step_gains"], masses)
        n = len(row["step_of_span"])
        assert built.omega[row["id"]] == pytest.approx([w[k] / n for k in row["step_of_span"]])
        assert sum(m * x for m, x in zip(masses, w)) == pytest.approx(n)


def test_spans_of_one_step_share_a_weight_only_in_the_step_unit(world):
    spans = provider(world, gain_unit="span")
    steps = provider(world, gain_unit="step")
    row = max(world["index"], key=lambda r: len(r["step_of_span"]))
    for omega, shared in ((steps.omega[row["id"]], True), (spans.omega[row["id"]], False)):
        by_step = {}
        for w, k in zip(omega, row["step_of_span"]):
            by_step.setdefault(k, set()).add(round(w, 12))
        assert all(len(v) == 1 for v in by_step.values()) == shared


def test_missing_files_stop_with_an_error_naming_the_file(world):
    root = world["root"]
    args = (str(root / "spans.jsonl"), world["data"])
    with pytest.raises(ValueError, match="teacher gain file"):
        SpanWeightProvider("alerd", *args, student_gain_file=str(root / "gain-student.json"), gain_source="teacher")
    with pytest.raises(ValueError, match="student gain file"):
        SpanWeightProvider("alerd", *args, teacher_gain_file=str(root / "gain-teacher.json"), gain_source="diff")
    with pytest.raises(ValueError, match="teacher gain file"):
        SpanWeightProvider("alerd", *args, student_gain_file=str(root / "gain-student.json"), gain_source="diff")
    with pytest.raises(FileNotFoundError, match="score_gain.sh"):
        SpanWeightProvider("alerd", *args, teacher_gain_file=str(root / "absent.json"), gain_source="teacher")


def test_unused_files_are_not_needed(world):
    root = world["root"]
    only_student = SpanWeightProvider("alerd", str(root / "spans.jsonl"), world["data"],
                                      student_gain_file=str(root / "gain-student.json"), gain_source="student")
    assert len(only_student.omega) == len(world["index"])


def test_gains_scored_on_another_text_stop_the_run(world, tmp_path):
    payload = json.loads((world["root"] / "gain-student.json").read_text())
    payload["samples"]["0"]["hash"] = "0" * 40
    bad = tmp_path / "gain-student.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        SpanWeightProvider("alerd", str(world["root"] / "spans.jsonl"), world["data"], student_gain_file=str(bad))


@pytest.mark.parametrize("kwargs", [dict(source="random"), dict(answer_only_unit=True), dict(normalization="none"),
                                    dict(normalization="corpus"), dict(tau=2.0), dict(gain_unit="step", source="random"),
                                    dict(gain_unit="step", answer_only_unit=True),
                                    dict(gain_unit="step", normalization="corpus")])
def test_variants_keep_positive_weights(world, kwargs):
    built = provider(world, **kwargs)
    for row in world["index"]:
        omega = built.omega[row["id"]]
        assert min(omega) > 0
        if kwargs.get("normalization", "per-trace") == "per-trace":
            assert sum(omega) == pytest.approx(1.0)


def test_corpus_normalization_preserves_the_corpus_budget(world):
    built = provider(world, normalization="corpus")
    total = sum(len(r["step_of_span"]) for r in world["index"])
    assert sum(sum(built.omega[r["id"]]) * len(r["step_of_span"]) for r in world["index"]) == pytest.approx(total)


def test_random_source_permutes_within_each_sample(world):
    real, shuffled = provider(world), provider(world, "random")
    for row in world["index"]:
        a, b = real.omega[row["id"]], shuffled.omega[row["id"]]
        assert sorted(a) == pytest.approx(sorted(b))  # same weights, other spans
    assert any(real.omega[r["id"]] != pytest.approx(shuffled.omega[r["id"]]) for r in world["index"])


def test_answer_only_unit_keeps_the_answer_step_at_weight_one(world):
    built = provider(world, answer_only_unit=True)
    for row in world["index"]:
        n = len(row["step_of_span"])
        last = [w for w, k in zip(built.omega[row["id"]], row["step_of_span"]) if k == row["n_steps"] - 1]
        assert last and all(w * n == pytest.approx(1.0) for w in last)  # the last step states the boxed answer


def test_predictability_scores_are_per_step_and_spans_inherit(world):
    easy, hard = provider(world, "predictability-easy"), provider(world, "predictability-hard")
    for row in world["index"]:
        assert sum(easy.omega[row["id"]]) == pytest.approx(1.0)
        by_step = {}
        for w, k in zip(easy.omega[row["id"]], row["step_of_span"]):
            by_step.setdefault(k, set()).add(round(w, 12))
        assert all(len(v) == 1 for v in by_step.values())
    assert easy.omega[0] != pytest.approx(hard.omega[0])


def test_hash_and_span_count_mismatches_stop_the_run(world):
    built = provider(world)
    row = world["index"][0]
    n = len(row["step_of_span"])
    built.check(row["id"], row["hash"], row["teacher_hash"], n)
    with pytest.raises(ValueError, match="hash"):
        built.check(row["id"], "other", row["teacher_hash"], n)
    with pytest.raises(ValueError, match="hash"):
        built.check(row["id"], row["hash"], "other", n)
    with pytest.raises(ValueError, match="spans"):
        built.check(row["id"], row["hash"], row["teacher_hash"], n + 1)
    with pytest.raises(ValueError, match="not in the span index"):
        built.check(99, row["hash"], row["teacher_hash"], n)


def test_batch_weights_pad_with_zeros(world):
    built = provider(world)
    a, b = world["index"][0], world["index"][1]
    ns = [len(a["step_of_span"]), len(b["step_of_span"])]
    out = built.batch_weights([a["id"], b["id"]], [a["hash"], b["hash"]], [a["teacher_hash"], b["teacher_hash"]], ns,
                              max(ns) + 3)
    assert out.shape == (2, max(ns) + 3) and (out[:, max(ns):] == 0).all()
    assert out.sum(-1).tolist() == pytest.approx([1.0, 1.0])


def test_misconfigurations_are_rejected(world):
    root = world["root"]
    args = (str(root / "spans.jsonl"), world["data"])
    with pytest.raises(ValueError):
        SpanWeightProvider("alerd", *args)
    with pytest.raises(ValueError):
        SpanWeightProvider("predictability-easy", *args)
    with pytest.raises(ValueError):
        provider(world, "random", normalization="corpus")
    with pytest.raises(ValueError):
        provider(world, "bogus")
    with pytest.raises(ValueError):
        provider(world, gain_source="bogus")
    with pytest.raises(ValueError):
        provider(world, gain_unit="bogus")
