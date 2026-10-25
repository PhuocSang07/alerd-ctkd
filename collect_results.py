r"""Collect sgl evaluation summaries into LaTeX tables laid out like Tables 2-5 of ALERD.

    python collect_results.py --results-dir results --table main --output tables/main.tex
    python collect_results.py --results-dir results --table ablations --output tables/ablations.tex

Reads <results-dir>/<tag>/summary.json written by sgl.eval.evaluate (see scripts/alerd_ctkd_eval.sh). Tags:
<student>-zeroshot, teacher, and <student>-<method>-s<seed> for every trained row (method names of
scripts/alerd_ctkd_train.sh). A trained row is the mean over the seeds found; the ablation table uses one seed
(--ablation-seed). A row without results is left out and reported on stderr. Within a block the best value of a column is
bold and the second best underlined (ties share the format); zero-shot and teacher rows are not ranked.
"""
import argparse
import json
import re
import sys
from pathlib import Path

COLUMNS = ("gsm8k", "math500")
HEADER = (("GSM8K", "gsm8k"), ("MATH500", "math500"), ("Average", None))

STUDENTS = {
    "gpt2-xl": "GPT2-XL (1.5B), full fine-tuning",
    "llama-3.2-1b": "Llama-3.2-1B-Instruct, full fine-tuning",
}

MAIN_ROWS = [  # (label, tag suffix, ranked)
    ("Zero-shot", "zeroshot", False),
    ("Teacher", None, False),
    ("SFT", "sft", True),
    ("SRA", "sra", True),
    ("SRA-uniform", "sra-uniform", True),
    (r"\textbf{ALERD-CTKD (ours)}", "alerd-ctkd", True),
]

ABLATION_BLOCKS = [  # (block title, rows of (label, tag suffix))
    ("Gain source (span unit)", [("Frozen student", "alerd-ctkd-gain-student"), ("Teacher", "alerd-ctkd-gain-teacher"),
                                 ("Teacher $-$ student", "alerd-ctkd-gain-diff")]),
    ("Gain unit (main gain source)", [("Per span", "alerd-ctkd"), ("Per step, spans inherit", "alerd-ctkd-unit-step")]),
    ("Span score", [("SRA attention", "sra"), (r"Answer-gain allocation ($\tau=1$)", "alerd-ctkd"),
                    ("Predictability -- easy", "alerd-ctkd-pred-easy"), ("Predictability -- hard", "alerd-ctkd-pred-hard"),
                    ("Random assignment", "alerd-ctkd-random"),
                    ("Answer-only steps at weight 1", "alerd-ctkd-answer-only-unit")]),
    ("Mass normalization", [("Per-trace", "alerd-ctkd"), ("Corpus-level", "alerd-ctkd-corpus"),
                            ("No normalization", "alerd-ctkd-nonorm")]),
    (r"Shrinkage $\lambda$, $\tau=2$", [(r"Uniform, $\lambda=0$", "alerd-ctkd-t2-l0"),
                                        (r"ALERD-CTKD ($\lambda=0.5$)", "alerd-ctkd-t2-l05"),
                                        (r"ALERD-CTKD ($\lambda=1$)", "alerd-ctkd-t2-l1")]),
    (r"Temperature $\tau$, $\lambda=0.5$", [(r"ALERD-CTKD ($\tau=1$)", "alerd-ctkd"),
                                            (r"ALERD-CTKD ($\tau=2$)", "alerd-ctkd-t2-l05"),
                                            (r"ALERD-CTKD ($\tau=4$)", "alerd-ctkd-t4-l05")]),
]


def read_summary(path):
    """[(pass@1, pass@3)] in percent per benchmark of one summary.json, or None when a benchmark is missing."""
    if not path.exists():
        return None
    by_benchmark = {row["benchmark"]: row for row in json.loads(path.read_text())}
    if any(name not in by_benchmark for name in COLUMNS):
        return None
    return [(100 * by_benchmark[name]["pass@1"], 100 * by_benchmark[name]["pass@3"]) for name in COLUMNS]


def with_average(cells):
    return cells + [(sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))]


def seed_tags(results_dir, tag, seeds=None):
    """Existing result directories <tag>-s<seed>, restricted to `seeds` when given."""
    found = []
    for path in sorted(Path(results_dir).glob(f"{tag}-s*")):
        match = re.fullmatch(re.escape(tag) + r"-s(\d+)", path.name)
        if match and (seeds is None or int(match.group(1)) in seeds):
            found.append(path.name)
    return found


def load_scores(results_dir, tag, seeds=None, per_seed=False):
    """[(pass@1, pass@3)] in percent per benchmark and their unweighted average, or None.

    per_seed=False reads <results-dir>/<tag>/summary.json; True averages the seeds of <tag>-s<seed> (the mean of each
    benchmark over the seeds that have a complete summary).
    """
    if not per_seed:
        cells = read_summary(Path(results_dir) / tag / "summary.json")
        return None if cells is None else with_average(cells)
    runs = [read_summary(Path(results_dir) / name / "summary.json") for name in seed_tags(results_dir, tag, seeds)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    mean = [(sum(r[i][0] for r in runs) / len(runs), sum(r[i][1] for r in runs) / len(runs))
            for i in range(len(COLUMNS))]
    return with_average(mean)


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
    lines = [rf"    \multicolumn{{{1 + 2 * len(HEADER)}}}{{l}}{{\textit{{{title}}}}} \\", ""]
    for i, (label, scores, _) in enumerate(rows):
        lines += [format_row(label, scores, marks, i), ""]
    return "\n".join(lines)


def table_wrapper(body, caption, label, first_column, tabcolsep):
    spans = "\n".join(rf"    \cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(HEADER)))
    heads = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for name, _ in HEADER)
    return "\n".join([
        r"\begin{table*}[t]", r"  \centering", r"  \small", rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"  \begin{{tabular}}{{{first_column}{'cc' * len(HEADER)}}}", r"    \toprule", f"    & {heads} \\\\", spans,
        "    Method" + " & p@1 & p@3" * len(HEADER) + r" \\", r"    \midrule", "", body,
        r"    \bottomrule", r"  \end{tabular}", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}", r"\end{table*}", ""])


def collect_main(results_dir, students, seeds=None):
    blocks, missing = [], []
    teacher = load_scores(results_dir, "teacher")
    for student in students:
        rows = []
        for label, suffix, ranked in MAIN_ROWS:
            tag = "teacher" if suffix is None else f"{student}-{suffix}"
            if suffix is None:
                scores = teacher
            else:
                scores = load_scores(results_dir, tag, seeds, per_seed=ranked)
                found = len(seed_tags(results_dir, tag, seeds)) if ranked else 1
                if ranked and scores is not None and found < len(seeds or ()):
                    print(f"{tag}: {found} of {len(seeds)} seeds", file=sys.stderr)
            if scores is None:
                missing.append(tag)
                continue
            rows.append((label, scores, ranked))
        blocks.append(render_block(f"{STUDENTS[student]}", rows))
    caption = ("Pass@1 and pass@3 (\\%) with $n=3$ samples on GSM8K test and MATH500; trained rows are means over seeds. "
               "Best per column and block in bold, second best underlined; zero-shot and teacher rows are not ranked.")
    return table_wrapper("\n".join(blocks), caption, "tab:main", "l", "3.2pt"), missing


def collect_ablations(results_dir, student, seed):
    blocks, missing = [], []
    for title, specs in ABLATION_BLOCKS:
        rows = []
        for label, suffix in specs:
            scores = load_scores(results_dir, f"{student}-{suffix}", {seed}, per_seed=True)
            if scores is None:
                missing.append(f"{student}-{suffix}-s{seed}")
                continue
            rows.append((label, scores, True))
        blocks.append(render_block(title, rows))
    caption = (f"Ablations on {STUDENTS[student]}, seed {seed}. Best per column and block in bold, "
               "second best underlined.")
    return table_wrapper("\n".join(blocks), caption, "tab:ablations", "p{4.3cm}", "4pt"), missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--table", choices=("main", "ablations"), default="main")
    parser.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="seeds averaged in the main table")
    parser.add_argument("--ablation-student", default="llama-3.2-1b", choices=list(STUDENTS))
    parser.add_argument("--ablation-seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    if args.table == "main":
        latex, missing = collect_main(args.results_dir, args.students, set(args.seeds))
    else:
        latex, missing = collect_ablations(args.results_dir, args.ablation_student, args.ablation_seed)
    for tag in missing:
        print(f"no results for {tag}", file=sys.stderr)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(latex, encoding="utf-8")
    else:
        print(latex)


if __name__ == "__main__":
    main()
