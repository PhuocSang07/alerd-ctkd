import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect_results import (ABLATION_BLOCKS, MAIN_ROWS, collect_ablations, collect_main, load_scores,  # noqa: E402
                             main, rank_marks, seed_tags)

STUDENT = "llama-3.2-1b"


def write_summary(root, tag, p1, p3=None):
    """p1 / p3: accuracy per benchmark in [0, 1] (gsm8k, math500)."""
    p3 = p3 or [min(1.0, x + 0.1) for x in p1]
    rows = [{"benchmark": b, "pass@1": a, "pass@3": c, "model": tag} for b, a, c in zip(("gsm8k", "math500"), p1, p3)]
    (root / tag).mkdir(parents=True, exist_ok=True)
    (root / tag / "summary.json").write_text(json.dumps(rows))


def test_load_scores_orders_columns_and_averages_in_percent(tmp_path):
    write_summary(tmp_path, "teacher", [0.80, 0.50], [0.90, 0.60])
    cells = load_scores(tmp_path, "teacher")
    assert cells[:2] == [pytest.approx((80.0, 90.0)), pytest.approx((50.0, 60.0))]  # GSM8K, MATH500
    assert cells[2] == pytest.approx((65.0, 75.0))
    assert load_scores(tmp_path, "missing") is None


def test_a_summary_without_one_of_the_benchmarks_is_ignored(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "summary.json").write_text(json.dumps([{"benchmark": "gsm8k", "pass@1": 0.5, "pass@3": 0.6}]))
    assert load_scores(tmp_path, "x") is None


def test_trained_rows_average_the_seeds_found(tmp_path):
    write_summary(tmp_path, f"{STUDENT}-sra-s42", [0.40, 0.10], [0.50, 0.20])
    write_summary(tmp_path, f"{STUDENT}-sra-s43", [0.60, 0.30], [0.70, 0.40])
    write_summary(tmp_path, f"{STUDENT}-sra-uniform-s42", [0.99, 0.99])  # another method, not a seed of sra
    assert seed_tags(tmp_path, f"{STUDENT}-sra") == [f"{STUDENT}-sra-s42", f"{STUDENT}-sra-s43"]
    both = load_scores(tmp_path, f"{STUDENT}-sra", per_seed=True)
    assert both[0] == pytest.approx((50.0, 60.0)) and both[1] == pytest.approx((20.0, 30.0))
    only_42 = load_scores(tmp_path, f"{STUDENT}-sra", {42}, per_seed=True)
    assert only_42[0] == pytest.approx((40.0, 50.0))
    assert load_scores(tmp_path, f"{STUDENT}-sra", {99}, per_seed=True) is None


def test_rank_marks_bold_best_underline_second_and_share_ties():
    scores = lambda v: [(v, v)] * 3  # noqa: E731
    marks = rank_marks([(0, scores(10.0)), (1, scores(30.0)), (2, scores(30.0)), (3, scores(20.0))])
    assert marks[(1, 0, 0)] == marks[(2, 0, 0)] == "best" and marks[(3, 0, 0)] == "second" and (0, 0, 0) not in marks


def fill_main(root, seeds=(42, 43, 44)):
    for method, base in (("sft", 0.30), ("sra", 0.40), ("sra-uniform", 0.41), ("alerd-ctkd", 0.45)):
        for seed in seeds:
            write_summary(root, f"{STUDENT}-{method}-s{seed}", [base + 0.2, base - 0.2])
    write_summary(root, f"{STUDENT}-zeroshot", [0.2, 0.1])
    write_summary(root, "teacher", [0.9, 0.7])


def test_main_table_has_the_layout_of_alerd_table_2(tmp_path):
    fill_main(tmp_path)
    latex, missing = collect_main(tmp_path, [STUDENT], {42, 43, 44})

    assert r"\small" in latex and r"\setlength{\tabcolsep}{3.2pt}" in latex and "resizebox" not in latex
    assert r"\begin{tabular}{lcccccc}" in latex and latex.count(r"\cmidrule(lr)") == 3
    assert "GSM8K" in latex and "MATH500" in latex and "AIME" not in latex
    assert r"\textit{Llama-3.2-1B-Instruct, full fine-tuning}" in latex
    assert r"\multicolumn{7}{l}" in latex
    assert [label for label, _, _ in MAIN_ROWS] == ["Zero-shot", "Teacher", "SFT", "SRA", "SRA-uniform",
                                                    r"\textbf{ALERD-CTKD (ours)}"]
    assert "ALERD " not in latex.replace("ALERD-CTKD", "")  # no black-box ALERD row
    assert r"\underline{" in latex and missing == []
    # 3 column groups of 2 numbers: 6 data cells, hence 6 ampersands, in every method row
    rows = [chunk for chunk in latex.split("\n\n") if chunk.strip().endswith("\\\\") and "multicolumn" not in chunk]
    assert len(rows) == 6 and all(chunk.count("&") == 6 for chunk in rows)


def test_best_trained_row_is_bold_and_the_ranking_ignores_zero_shot_and_teacher(tmp_path):
    fill_main(tmp_path)
    write_summary(tmp_path, f"{STUDENT}-zeroshot", [0.99, 0.99])
    latex, _ = collect_main(tmp_path, [STUDENT], {42, 43, 44})
    teacher_row = latex.split("Teacher")[1].split(r"\\")[0]
    assert r"\textbf" not in teacher_row and r"\underline" not in teacher_row
    zero_row = latex.split("Zero-shot")[1].split(r"\\")[0]
    assert r"\textbf" not in zero_row
    ours = latex.split(r"\textbf{ALERD-CTKD (ours)}")[1].split(r"\\")[0]
    assert ours.count(r"\textbf") == 6  # best in every column of the ranked rows


def test_missing_rows_are_reported_and_partial_seeds_noted(tmp_path, capsys):
    write_summary(tmp_path, f"{STUDENT}-sra-s42", [0.3, 0.2])
    write_summary(tmp_path, "teacher", [0.9, 0.7])
    _, missing = collect_main(tmp_path, [STUDENT], {42, 43, 44})
    assert f"{STUDENT}-alerd-ctkd" in missing and f"{STUDENT}-zeroshot" in missing
    assert f"{STUDENT}-sra: 1 of 3 seeds" in capsys.readouterr().err


def test_ablation_blocks_cover_gain_source_unit_and_the_alerd_tables(tmp_path):
    for method, base in (("sra", 0.3), ("alerd-ctkd", 0.5), ("alerd-ctkd-random", 0.4), ("alerd-ctkd-t2-l0", 0.35),
                         ("alerd-ctkd-gain-diff", 0.55), ("alerd-ctkd-unit-step", 0.45)):
        write_summary(tmp_path, f"{STUDENT}-{method}-s42", [base, base])
    write_summary(tmp_path, f"{STUDENT}-sra-s43", [0.9, 0.9])  # another seed must not enter an ablation
    latex, missing = collect_ablations(tmp_path, STUDENT, 42)
    for title in ("Gain source", "Gain unit", "Span score", "Mass normalization", "Shrinkage", "Temperature"):
        assert title in latex
    assert [t for t, _ in ABLATION_BLOCKS][:2] == ["Gain source (span unit)", "Gain unit (main gain source)"]
    assert r"\begin{tabular}{p{4.3cm}cccccc}" in latex and r"\setlength{\tabcolsep}{4pt}" in latex
    assert f"{STUDENT}-alerd-ctkd-corpus-s42" in missing and f"{STUDENT}-alerd-ctkd-gain-teacher-s42" in missing
    assert "90.00" not in latex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tables_compile(tmp_path):
    fill_main(tmp_path)
    latex, _ = collect_main(tmp_path, [STUDENT], {42, 43, 44})
    doc = tmp_path / "t.tex"
    doc.write_text("\\documentclass{article}\\usepackage[landscape]{geometry}\\usepackage{booktabs}\\begin{document}\n"
                   + latex + "\\end{document}")
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0


def test_cli_writes_the_file(tmp_path):
    write_summary(tmp_path / "res", "gpt2-xl-sra-s42", [0.5, 0.1])
    out = tmp_path / "tables" / "main.tex"
    main(["--results-dir", str(tmp_path / "res"), "--students", "gpt2-xl", "--output", str(out)])
    assert out.read_text(encoding="utf-8").startswith(r"\begin{table*}")
    abl = tmp_path / "tables" / "ablations.tex"
    main(["--results-dir", str(tmp_path / "res"), "--table", "ablations", "--ablation-student", "gpt2-xl",
          "--output", str(abl)])
    assert "Gain source" in abl.read_text(encoding="utf-8")
