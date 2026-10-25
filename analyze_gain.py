r"""Statistics of the span gains of the student, the teacher and their difference (run after scripts/score_gain.sh).

    python analyze_gain.py --spans work/<student>/spans-<teacher>.jsonl --data work/<student>/train-sra.jsonl \
        --student-gain work/<student>/gain-tokens-<student>.json --teacher-gain work/<teacher>/gain-tokens-<teacher>.json \
        --output plans/reports/analysis-gain.md

Reports, for each gain source (student, teacher, teacher - student): the mean span gain by position decile, the share of
samples whose largest-gain span falls in each type (number, operator, connector, word, final answer), and the within-sample
Spearman correlation between the student's and the teacher's span gains. Span gains are sums of token gains (token_gain.py).
"""
import argparse
import statistics
from pathlib import Path

import numpy as np

from analyze_weights import SPAN_TYPES, spearman, span_types
from span_weights import load_jsonl
from token_gain import GainFile, span_gains

SOURCES = ("student", "teacher", "teacher - student")
DECILES = 10


def sample_gains(row, student, teacher):
    """{source: span gains} of one sample, checked against the span index like the trainer does."""
    s_entry = student.entry(row["id"], row["hash"], row["student_tokens"][0][0], row["student_tokens"][-1][1])
    t_entry = teacher.entry(row["id"], row["teacher_hash"], row["teacher_tokens"][0][0], row["teacher_tokens"][-1][1])
    s, t = span_gains(s_entry, row["student_tokens"]), span_gains(t_entry, row["teacher_tokens"])
    return {"student": s, "teacher": t, "teacher - student": [b - a for a, b in zip(s, t)]}


def build_report(spans, examples, student, teacher):
    by_position = {s: [[] for _ in range(DECILES)] for s in SOURCES}
    top_type = {s: {t: 0 for t in SPAN_TYPES} for s in SOURCES}
    correlations, n_samples = [], 0
    for row in spans:
        gains = sample_gains(row, student, teacher)
        types = span_types(examples[row["id"]], row)
        n = len(types)
        n_samples += 1
        for source in SOURCES:
            values = gains[source]
            for i, v in enumerate(values):
                by_position[source][min(DECILES - 1, i * DECILES // n)].append(v)
            top_type[source][types[int(np.argmax(values))]] += 1
        rho = spearman(gains["student"], gains["teacher"]) if n >= 4 else None
        if rho is not None:
            correlations.append(rho)
    lines = ["# Span gain statistics", "", f"Samples: {n_samples}.", "",
             "## Mean span gain (log-probability of the answer) by position decile of the response", "",
             "| Source | " + " | ".join(f"{10 * i}-{10 * (i + 1)}%" for i in range(DECILES)) + " |",
             "|---|" + "---|" * DECILES]
    for source in SOURCES:
        lines.append(f"| {source} | " + " | ".join(f"{statistics.fmean(d):.3f}" if d else "n/a"
                                                    for d in by_position[source]) + " |")
    lines += ["", "## Type of the span with the largest gain (share of samples)", "",
              "| Source | " + " | ".join(SPAN_TYPES) + " |", "|---|" + "---|" * len(SPAN_TYPES)]
    for source in SOURCES:
        lines.append(f"| {source} | " + " | ".join(f"{100 * top_type[source][t] / n_samples:.1f}%"
                                                   for t in SPAN_TYPES) + " |")
    lines += ["", "## Student versus teacher span gains", ""]
    if correlations:
        lines.append(f"Within-sample Spearman correlation: median {statistics.median(correlations):.3f}, "
                     f"mean {statistics.fmean(correlations):.3f} over {len(correlations)} samples.")
    else:
        lines.append("No sample with at least 4 spans and non-constant gains.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--student-gain", required=True)
    parser.add_argument("--teacher-gain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    examples = {r["id"]: r for r in load_jsonl(args.data)}
    report = build_report(load_jsonl(args.spans), examples, GainFile(args.student_gain, "student"),
                          GainFile(args.teacher_gain, "teacher"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
