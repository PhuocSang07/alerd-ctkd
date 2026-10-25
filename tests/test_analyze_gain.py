import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from analyze_gain import build_report, main, sample_gains  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from span_index import build_rows  # noqa: E402
from token_gain import GainFile, span_gains  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = random.Random(7)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student, teacher = train_tokenizer("bpe", corpus), train_tokenizer("bpe", corpus, vocab_size=420)
    rows = make_rows(rng, 10)
    root = tmp_path_factory.mktemp("again")
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index = build_rows(LLMDataset(str(root / "train.jsonl"), student, teacher), student, teacher, 256)
    (root / "spans.jsonl").write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
    for side, tok in (("student", student), ("teacher", teacher)):
        (root / f"gain-{side}.json").write_text(json.dumps(make_gain_file(rows, tok, side, rng)))
    return {"root": root, "rows": rows, "index": index}


def test_sample_gains_has_the_three_sources_and_diff_is_their_difference(world):
    student = GainFile(str(world["root"] / "gain-student.json"), "student")
    teacher = GainFile(str(world["root"] / "gain-teacher.json"), "teacher")
    row = world["index"][0]
    gains = sample_gains(row, student, teacher)
    assert len(gains["student"]) == len(gains["teacher"]) == len(row["student_tokens"])
    assert gains["teacher - student"] == pytest.approx([t - s for s, t in zip(gains["student"], gains["teacher"])])
    entry = student.entry(row["id"], row["hash"], row["student_tokens"][0][0], row["student_tokens"][-1][1])
    assert gains["student"] == pytest.approx(span_gains(entry, row["student_tokens"]))


def test_cli_report_covers_position_type_and_correlation(world, tmp_path):
    root = world["root"]
    main(["--spans", str(root / "spans.jsonl"), "--data", str(root / "train.jsonl"),
          "--student-gain", str(root / "gain-student.json"), "--teacher-gain", str(root / "gain-teacher.json"),
          "--output", str(tmp_path / "gain.md")])
    report = (tmp_path / "gain.md").read_text(encoding="utf-8")
    assert "Samples: 10" in report and "teacher - student" in report and "position decile" in report
    assert "Within-sample Spearman correlation: median" in report
    assert "| final answer |" in report  # a column of the span-type table
