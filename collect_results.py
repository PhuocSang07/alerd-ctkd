r"""Collect sgl evaluation summaries into LaTeX tables laid out like Tables 2-5 of ALERD.

    python collect_results.py --results-dir results --table main --output tables/main.tex
    python collect_results.py --results-dir results --table ablations --output tables/ablations.tex

Reads <results-dir>/<tag>/summary.json written by sgl.eval.evaluate (see scripts/alerd_ctkd_eval.sh). Tags:
<student>-zeroshot, teacher, <student>-<method> with the method names of scripts/alerd_ctkd_train.sh, plus
<student>-sft and <student>-alerd for the black-box rows trained with sgl. A row without results is left out and
reported on stderr. Zero-shot falls back to the numbers printed in ALERD Table 2. Within a block the best value of
a column is bold and the second best underlined (ties share the format); zero-shot and teacher rows are not ranked.
"""
import argparse
import json
import sys
from pathlib import Path

COLUMNS = ("aime25", "aime24", "amc12", "math500")
HEADER = (("AIME25", "aime25"), ("AIME24", "aime24"), ("AMC12", "amc12"), ("MATH500", "math500"), ("Average", None))

STUDENTS = {
    "r1-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B, full fine-tuning",
    "qwen25-7b": "Qwen2.5-7B-Instruct, LoRA",
    "qwen3-8b": "Qwen3-8B, LoRA",
}

# (pass@1, pass@3) per column then average, in percent: the Zero-shot rows of ALERD Table 2
ZERO_SHOT_TABLE_2 = {
    "r1-qwen-1.5b": [(10.00, 20.00), (15.56, 33.33), (49.80, 63.86), (72.33, 81.80), (36.92, 49.75)],
    "qwen25-7b": [(4.44, 10.00), (8.89, 13.33), (44.98, 63.86), (74.67, 85.20), (33.24, 43.10)],
    "qwen3-8b": [(20.00, 23.33), (22.22, 33.33), (52.61, 68.67), (77.00, 84.80), (42.96, 52.53)],
}

MAIN_ROWS = [  # (label, tag suffix, ranked)
    ("Zero-shot", "zeroshot", False),
    ("Teacher", None, False),
    ("SFT", "sft", True),
    ("ALERD", "alerd", True),
    ("SRA", "sra", True),
    ("SRA-uniform", "sra-uniform", True),
    (r"\textbf{ALERD-CTKD (ours)}", "alerd-ctkd", True),
]

ABLATION_BLOCKS = [  # (block title, rows of (label, tag suffix))
    ("Step score", [("SRA attention", "sra"), (r"Answer-gain allocation ($\tau=1$)", "alerd-ctkd"),
                    ("Predictability -- easy", "alerd-ctkd-pred-easy"), ("Predictability -- hard", "alerd-ctkd-pred-hard"),
                    ("Random assignment", "alerd-ctkd-random"),
                    ("Answer-only steps at weight 1", "alerd-ctkd-answer-only-unit")]),
    ("Token-mass normalization", [("Per-trace", "alerd-ctkd"), ("Corpus-level", "alerd-ctkd-corpus"),
                                  ("No normalization", "alerd-ctkd-nonorm")]),
    (r"Shrinkage $\lambda$, $\tau=2$", [(r"Uniform, $\lambda=0$", "alerd-ctkd-t2-l0"),
                                        (r"ALERD-CTKD ($\lambda=0.5$)", "alerd-ctkd-t2-l05"),
                                        (r"ALERD-CTKD ($\lambda=1$)", "alerd-ctkd-t2-l1")]),
    (r"Temperature $\tau$, $\lambda=0.5$", [(r"ALERD-CTKD ($\tau=1$)", "alerd-ctkd"),
                                            (r"ALERD-CTKD ($\tau=2$)", "alerd-ctkd-t2-l05"),
                                            (r"ALERD-CTKD ($\tau=4$)", "alerd-ctkd-t4-l05")]),
]


def load_scores(results_dir, tag):
    """[(pass@1, pass@3)] in percent for the four benchmarks and their unweighted average, or None."""
    path = Path(results_dir) / tag / "summary.json"
    if not path.exists():
        return None
    by_benchmark = {row["benchmark"]: row for row in json.loads(path.read_text())}
    if any(name not in by_benchmark for name in COLUMNS):
        return None
    cells = [(100 * by_benchmark[name]["pass@1"], 100 * by_benchmark[name]["pass@3"]) for name in COLUMNS]
    average = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
    return cells + [average]


def rank_marks(rows):
    """{(row index, metric index, column): 'best' | 'second'} over the ranked rows (compared at two decimals)."""
    marks = {}
    for column in range(len(HEADER)):
        for metric in range(2):
            values = {i: round(scores[column][metric], 2) for i, scores in rows}
            distinct = sorted(set(values.values()), reverse=True)
            for i, value in values.items():
                if len(rows) > 1 and value == distinct[0]:
                    marks[(i, metric, column)] = "best"
                elif len(distinct) > 1 and value == distinct[1]:
                    marks[(i, metric, column)] = "second"
    return marks


def format_cell(value, mark):
    text = f"{value:.2f}"
    if mark == "best":
        return rf"\textbf{{{text}}}"
    if mark == "second":
        return rf"\underline{{{text}}}"
    return text


def format_row(label, scores, marks, index):
    parts = []
    for column in range(len(HEADER)):
        for metric in range(2):
            parts.append(format_cell(scores[column][metric], marks.get((index, metric, column))))
    cells = [" & ".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
    return f"    {label}\n    & " + "\n    & ".join(cells) + r" \\"


def render_block(title, rows):
    """rows: [(label, scores, ranked)]; the block title is a full-width italic line."""
    ranked = [(i, scores) for i, (_, scores, is_ranked) in enumerate(rows) if is_ranked]
    marks = rank_marks(ranked) if ranked else {}
    lines = [rf"    \multicolumn{{11}}{{l}}{{\textit{{{title}}}}} \\", ""]
    for i, (label, scores, _) in enumerate(rows):
        lines += [format_row(label, scores, marks, i), ""]
    return "\n".join(lines)


def table_wrapper(body, caption, label, column_spec, tabcolsep):
    spans = "\n".join(rf"    \cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(HEADER)))
    heads = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for name, _ in HEADER)
    return "\n".join([
        r"\begin{table*}[t]", r"  \centering", r"  \small", rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"  \begin{{tabular}}{{{column_spec}}}", r"    \toprule", f"    & {heads} \\\\", spans,
        "    Method" + " & p@1 & p@3" * len(HEADER) + r" \\", r"    \midrule", "", body,
        r"    \bottomrule", r"  \end{tabular}", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}", r"\end{table*}", ""])


def collect_main(results_dir, students):
    blocks, missing = [], []
    teacher = load_scores(results_dir, "teacher")
    for student in students:
        rows = []
        for label, suffix, ranked in MAIN_ROWS:
            tag = "teacher" if suffix is None else f"{student}-{suffix}"
            scores = teacher if suffix is None else load_scores(results_dir, tag)
            if scores is None and suffix == "zeroshot":
                scores = ZERO_SHOT_TABLE_2[student]
            if scores is None:
                missing.append(tag)
                continue
            rows.append((label, scores, ranked))
        blocks.append(render_block(f"{STUDENTS[student]}", rows))
    caption = ("Pass@1 and pass@3 (\\%) with $n=3$ samples. Best per column and block in bold, second best underlined; "
               "zero-shot and teacher rows are not ranked.")
    body = "\n".join(blocks)
    return table_wrapper(body, caption, "tab:main", "lcccccccccc", "3.2pt"), missing


def collect_ablations(results_dir, student):
    blocks, missing = [], []
    for title, specs in ABLATION_BLOCKS:
        rows = []
        for label, suffix in specs:
            scores = load_scores(results_dir, f"{student}-{suffix}")
            if scores is None:
                missing.append(f"{student}-{suffix}")
                continue
            rows.append((label, scores, True))
        blocks.append(render_block(title, rows))
    caption = f"Ablations on {STUDENTS[student]}. Best per column and block in bold, second best underlined."
    return table_wrapper("\n".join(blocks), caption, "tab:ablations", "p{4.3cm}cccccccccc", "4pt"), missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--table", choices=("main", "ablations"), default="main")
    parser.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    parser.add_argument("--ablation-student", default="r1-qwen-1.5b", choices=list(STUDENTS))
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    if args.table == "main":
        latex, missing = collect_main(args.results_dir, args.students)
    else:
        latex, missing = collect_ablations(args.results_dir, args.ablation_student)
    for tag in missing:
        print(f"no results for {tag}", file=sys.stderr)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(latex, encoding="utf-8")
    else:
        print(latex)


if __name__ == "__main__":
    main()
