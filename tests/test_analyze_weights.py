import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_weights import (answer_step, build_report, correlation_summary, main, ranks,  # noqa: E402
                             spearman)


def test_ranks_average_ties_and_spearman_matches_known_values():
    assert ranks([10, 20, 20, 30]).tolist() == [0.0, 1.5, 1.5, 3.0]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    scipy_stats = pytest.importorskip("scipy.stats")
    a, b = np.random.default_rng(0).normal(size=(2, 30))
    assert spearman(a, b) == pytest.approx(scipy_stats.spearmanr(a, b)[0])


def make_dump(n_examples=12, n_spans=20, seed=0):
    rng = random.Random(seed)
    dump, spans, examples = [], {}, {}
    for rid in range(n_examples):
        attention = [rng.random() for _ in range(n_spans)]
        omega = [a + 0.05 * rng.random() for a in attention] if rid % 2 == 0 else [rng.random() for _ in range(n_spans)]
        attention = [a / sum(attention) for a in attention]
        omega = [o / sum(omega) for o in omega]
        dump += [{"id": rid, "attention": attention, "omega": omega}] * 2  # repeated across epochs
        steps = sorted(rng.choices(range(4), k=n_spans))
        spans[rid] = {"id": rid, "step_of_span": steps, "char_ends": [5 * (i + 1) for i in range(n_spans)]}
        text = "".join(f"s{i:03d} " for i in range(n_spans))
        examples[rid] = {"id": rid, "prompt": "", "output": text, "step_char_ends": [25, 50, 75, 100]}
    return dump, spans, examples


def test_correlation_summary_separates_aligned_from_random_weights():
    dump, _, _ = make_dump()
    unique = {r["id"]: r for r in dump}.values()
    aligned = [r for r in unique if r["id"] % 2 == 0]
    random_pairs = [r for r in unique if r["id"] % 2 == 1]
    assert correlation_summary(aligned)["median"] > 0.9
    summary = correlation_summary(random_pairs)
    assert abs(summary["median"]) < 0.5 and summary["n"] == len(random_pairs)
    assert correlation_summary(aligned)["outside_95"] == 1.0


def test_answer_step_finds_the_step_that_states_the_answer():
    example = {"output": "a b c so \\boxed{42} end", "step_char_ends": [6, 20, 23]}
    assert answer_step(example) == 1
    assert answer_step({"output": "no answer", "step_char_ends": [9]}) is None


def test_report_deduplicates_epochs_and_requires_omega():
    dump, spans, examples = make_dump()
    report, rows = build_report(dump, spans, examples)
    assert len(rows) == 12 and "Within-example Spearman" in report and "Step stating the answer" in report
    with pytest.raises(ValueError):
        build_report([{"id": 0, "attention": [0.5, 0.5]}], spans, examples)


def test_cli_writes_report_and_coloured_example(tmp_path):
    dump, spans, examples = make_dump()
    for name, payload in (("dump", dump), ("spans", list(spans.values())), ("data", list(examples.values()))):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in payload))
    main(["--dump", str(tmp_path / "dump.jsonl"), "--spans", str(tmp_path / "spans.jsonl"),
          "--data", str(tmp_path / "data.jsonl"), "--output", str(tmp_path / "report.md"),
          "--html", str(tmp_path / "example.html")])
    assert "ALERD omega versus SRA attention" in (tmp_path / "report.md").read_text(encoding="utf-8")
    page = (tmp_path / "example.html").read_text(encoding="utf-8")
    assert page.count("<span style") == 2 * 20 and "SRA attention" in page and "ALERD omega" in page


def test_span_types_follow_digits_symbols_connectors_and_the_final_step():
    example = {"step_char_ends": [10, 20]}
    span_row = {"step_of_span": [0, 0, 0, 0, 1, 1]}
    pieces = ["She sold ", "48", " /", " so", " clips", " 72"]
    from analyze_weights import span_type, type_profile, type_table

    assert [span_type(p, step == 1) for p, step in zip(pieces, span_row["step_of_span"])] == [
        "word", "number", "operator", "connector", "final answer", "final answer"]
    weights = [0.1, 0.4, 0.1, 0.1, 0.2, 0.1]
    profile = type_profile([(["word", "number", "operator", "connector", "final answer", "final answer"], weights)])
    assert profile["number"] == (pytest.approx(1 / 6), pytest.approx(6 * 0.4))
    assert profile["final answer"][1] == pytest.approx(6 * 0.15)
    assert sum(share for share, _ in profile.values()) == pytest.approx(1.0)
    table = type_table({"a": profile, "b": profile})
    assert table[0].startswith("| Span type |") and len(table) == 2 + 5 and "n/a" not in "\n".join(table)
    assert example and span_row


def test_report_lists_the_weight_by_span_type():
    dump, spans, examples = make_dump()
    report, _ = build_report(dump, spans, examples)
    assert "Weight by span type" in report and "| number |" in report and "| final answer |" in report
