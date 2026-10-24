import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect_results import (ZERO_SHOT_TABLE_2, collect_ablations, collect_main, load_scores,  # noqa: E402
                             main, rank_marks)


def write_summary(root, tag, p1, p3=None):
    """p1 / p3: accuracy per benchmark in [0, 1] (math500, aime24, aime25, amc12)."""
    p3 = p3 or [min(1.0, x + 0.1) for x in p1]
    rows = [{"benchmark": b, "pass@1": a, "pass@3": c, "model": tag}
            for b, a, c in zip(("math500", "aime24", "aime25", "amc12"), p1, p3)]
    (root / tag).mkdir(parents=True)
    (root / tag / "summary.json").write_text(json.dumps(rows))


def test_load_scores_orders_columns_and_averages_in_percent(tmp_path):
    write_summary(tmp_path, "s-sra", [0.70, 0.20, 0.10, 0.50], [0.80, 0.30, 0.20, 0.60])
    cells = load_scores(tmp_path, "s-sra")
    assert cells[:4] == [pytest.approx((10.0, 20.0)), pytest.approx((20.0, 30.0)),
                         pytest.approx((50.0, 60.0)), pytest.approx((70.0, 80.0))]  # AIME25, AIME24, AMC12, MATH500
    assert cells[4] == pytest.approx((37.5, 47.5))
    assert load_scores(tmp_path, "missing") is None


def test_rank_marks_bold_best_underline_second_and_share_ties():
    scores = lambda v: [(v, v)] * 5  # noqa: E731
    marks = rank_marks([(0, scores(10.0)), (1, scores(30.0)), (2, scores(30.0)), (3, scores(20.0))])
    assert marks[(1, 0, 0)] == marks[(2, 0, 0)] == "best" and marks[(3, 0, 0)] == "second" and (0, 0, 0) not in marks


def test_main_table_has_the_layout_of_alerd_table_2(tmp_path):
    for method, base in (("sft", 0.30), ("sra", 0.40), ("sra-uniform", 0.41), ("alerd-ctkd", 0.45)):
        write_summary(tmp_path, f"r1-qwen-1.5b-{method}", [base + 0.2, base - 0.2, base - 0.25, base])
    write_summary(tmp_path, "teacher", [0.9, 0.7, 0.6, 0.8])
    latex, missing = collect_main(tmp_path, ["r1-qwen-1.5b"])

    assert r"\small" in latex and r"\setlength{\tabcolsep}{3.2pt}" in latex and "resizebox" not in latex
    assert r"\begin{tabular}{lcccccccccc}" in latex and latex.count(r"\cmidrule(lr)") == 5
    assert r"\textit{DeepSeek-R1-Distill-Qwen-1.5B, full fine-tuning}" in latex
    assert "10.00 & 20.00" in latex  # zero-shot falls back to ALERD Table 2
    assert r"\textbf{ALERD-CTKD (ours)}" in latex and latex.count(r"\textbf{") > 1
    assert r"\underline{" in latex
    assert sorted(missing) == ["r1-qwen-1.5b-alerd"]
    # 5 column groups of 2 numbers: 10 data cells, hence 10 ampersands, in every method row
    rows = [chunk for chunk in latex.split("\n\n") if chunk.strip().endswith("\\\\") and "multicolumn" not in chunk]
    assert len(rows) == 6 and all(chunk.count("&") == 10 for chunk in rows)


def test_zero_shot_and_teacher_rows_are_not_ranked(tmp_path):
    write_summary(tmp_path, "r1-qwen-1.5b-sra", [0.2, 0.2, 0.2, 0.2])
    write_summary(tmp_path, "r1-qwen-1.5b-sra-uniform", [0.1, 0.1, 0.1, 0.1])
    write_summary(tmp_path, "teacher", [0.99, 0.99, 0.99, 0.99])
    latex, _ = collect_main(tmp_path, ["r1-qwen-1.5b"])
    teacher_row = latex.split("Teacher")[1].split(r"\\")[0]
    assert r"\textbf" not in teacher_row and r"\underline" not in teacher_row
    sra_row = latex.split("SRA\n")[1].split(r"\\")[0]
    assert sra_row.count(r"\textbf") == 10  # alone among the ranked rows it is best everywhere


def test_zero_shot_constants_are_complete():
    assert set(ZERO_SHOT_TABLE_2) == {"r1-qwen-1.5b", "qwen25-7b", "qwen3-8b"}
    assert all(len(v) == 5 for v in ZERO_SHOT_TABLE_2.values())


def test_ablation_table_blocks_and_missing_rows(tmp_path):
    for method, base in (("sra", 0.3), ("alerd-ctkd", 0.5), ("alerd-ctkd-random", 0.4), ("alerd-ctkd-t2-l0", 0.35)):
        write_summary(tmp_path, f"r1-qwen-1.5b-{method}", [base] * 4)
    latex, missing = collect_ablations(tmp_path, "r1-qwen-1.5b")
    for title in ("Step score", "Token-mass normalization", "Shrinkage", "Temperature"):
        assert title in latex
    assert r"\begin{tabular}{p{4.3cm}cccccccccc}" in latex and r"\setlength{\tabcolsep}{4pt}" in latex
    assert "r1-qwen-1.5b-alerd-ctkd-corpus" in missing


def test_cli_writes_the_file(tmp_path):
    write_summary(tmp_path / "res", "qwen3-8b-sra", [0.5] * 4)
    out = tmp_path / "tables" / "main.tex"
    main(["--results-dir", str(tmp_path / "res"), "--students", "qwen3-8b", "--output", str(out)])
    assert out.read_text(encoding="utf-8").startswith(r"\begin{table*}")
