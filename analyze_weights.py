"""Compare ALERD span weights with SRA's attention weights (Figure 2 of ALERD, adapted to spans).

    python analyze_weights.py --dump outputs/<student>/alerd-ctkd/span_weights.jsonl \
        --spans work/<student>/spans-<teacher>.jsonl --data work/<student>/train-sra.jsonl \
        --output plans/reports/analysis-<timestamp>-alerd-ctkd.md --html plans/reports/example-weights.html

The dump comes from training with --dump_span_weights (DUMP_SPAN_WEIGHTS=1 in scripts/alerd_ctkd_train.sh) and holds,
per example, the teacher's normalised attention mass of each span and omega. The report gives
  1. the within-example Spearman correlation between omega and attention, against two independent random rankings;
  2. where SRA's attention goes: the first span, the top 10% of spans, and the step that states the answer;
  3. one example with spans coloured by both weights (HTML).
"""
import argparse
import html
import json
import random
import re
import statistics
from pathlib import Path

import numpy as np

from allocation import contains_answer, last_boxed

NULL_DRAWS = 2000
SPAN_TYPES = ("number", "operator", "connector", "word", "final answer")
CONNECTORS = {"so", "then", "therefore", "thus", "and", "but", "because", "since", "if", "now", "total", "altogether",
              "each", "of", "per", "in", "for", "after", "before", "more", "less", "than", "from", "with"}


def ranks(values):
    """Average ranks (ties share the mean rank)."""
    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="mergesort")
    out = np.empty(len(values))
    out[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        mask = values == value
        out[mask] = out[mask].mean()
    return out


def spearman(a, b):
    """Spearman correlation, or None when either side is constant."""
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def null_interval(n, rng, cache={}):
    """Central 95% range of the correlation between two independent random rankings of n items."""
    if n not in cache:
        draws = [spearman(rng.permutation(n), rng.permutation(n)) for _ in range(NULL_DRAWS)]
        cache[n] = tuple(np.percentile([d for d in draws if d is not None], [2.5, 97.5]))
    return cache[n]


def correlation_summary(rows, seed=0):
    rng = np.random.default_rng(seed)
    rhos, random_rhos, outside = [], [], 0
    for row in rows:
        rho = spearman(row["omega"], row["attention"])
        if rho is None or len(row["omega"]) < 4:
            continue
        rhos.append(rho)
        random_rhos.append(spearman(rng.permutation(len(row["omega"])), rng.permutation(len(row["omega"]))))
        low, high = null_interval(len(row["omega"]), rng)
        outside += not low <= rho <= high
    return {"n": len(rhos), "median": statistics.median(rhos), "mean": statistics.fmean(rhos),
            "random_median": statistics.median(r for r in random_rhos if r is not None),
            "outside_95": outside / len(rhos)}


def answer_step(example):
    """Index of the step whose text contains the gold answer (the last boxed answer), or None."""
    answer = last_boxed(example["output"])
    if not answer:
        return None
    starts = [0] + example["step_char_ends"][:-1]
    hits = [k for k, (a, b) in enumerate(zip(starts, example["step_char_ends"])) if contains_answer(example["output"][a:b], answer)]
    return hits[-1] if hits else None


def concentration(rows, spans, examples, key):
    """Mean share of the weight on the first span, on the top 10% spans, and on the answer-bearing step."""
    first, top, answer = [], [], []
    for row in rows:
        w = np.asarray(row[key], dtype=float)
        w = w / w.sum()
        first.append(w[0])
        top.append(np.sort(w)[::-1][: max(1, round(0.1 * len(w)))].sum())
        step = answer_step(examples[row["id"]])
        if step is not None:
            steps = np.asarray(spans[row["id"]]["step_of_span"])
            answer.append(w[steps == step].sum())
    return {"first_span": statistics.fmean(first), "top10": statistics.fmean(top),
            "answer_step": statistics.fmean(answer) if answer else None}


def span_pieces(example, span_row):
    """Text of each span of one example."""
    text = example["prompt"] + example["output"]
    ends = [len(example["prompt"])] + span_row["char_ends"]
    return [text[a:b] for a, b in zip(ends, ends[1:])]


def span_type(piece, in_final_step):
    """number (has a digit) | operator (symbols only) | connector (small closed word list) | word; the final-answer
    sentence is its own type."""
    if in_final_step:
        return "final answer"
    if re.search(r"\d", piece):
        return "number"
    word = piece.strip().lower().strip(".,:;!?")
    if word and not re.search(r"[A-Za-z]", word):
        return "operator"
    return "connector" if word in CONNECTORS else "word"


def span_types(example, span_row):
    last_step = len(example["step_char_ends"]) - 1
    return [span_type(piece, step == last_step) for piece, step in zip(span_pieces(example, span_row),
                                                                       span_row["step_of_span"])]


def type_profile(items):
    """items: [(types, weights)] per example, weights summing to 1. Returns {type: (share of spans, relative weight)}:
    the relative weight is the mean of N * weight over the spans of that type, 1 meaning uniform."""
    counts = {t: 0 for t in SPAN_TYPES}
    relative = {t: 0.0 for t in SPAN_TYPES}
    total = 0
    for types, weights in items:
        n = len(types)
        total += n
        for t, w in zip(types, weights):
            counts[t] += 1
            relative[t] += n * w
    return {t: (counts[t] / total, relative[t] / counts[t] if counts[t] else None) for t in SPAN_TYPES}


def type_table(profiles):
    """Markdown rows for {name: profile}: share of spans, then the relative weight of each weight source."""
    names = list(profiles)
    lines = ["| Span type | Share of spans | " + " | ".join(f"{n} (x uniform)" for n in names) + " |",
             "|---|---|" + "---|" * len(names)]
    first = profiles[names[0]]
    for t in SPAN_TYPES:
        cells = ["n/a" if profiles[n][t][1] is None else f"{profiles[n][t][1]:.2f}" for n in names]
        lines.append(f"| {t} | {100 * first[t][0]:.1f}% | " + " | ".join(cells) + " |")
    return lines


def example_html(example, span_row, row):
    """Spans of one example as two coloured rows: SRA attention and omega, shading = rank within the example."""
    pieces = [html.escape(piece) for piece in span_pieces(example, span_row)]

    def line(title, weights):
        shade = ranks(weights) / max(1, len(weights) - 1)
        body = "".join(f'<span style="background:rgba(220,60,40,{0.08 + 0.7 * s:.2f})" title="{w:.4f}">{p}</span>'
                       for p, w, s in zip(pieces, weights, shade))
        return f"<h3>{title}</h3><div style=\"white-space:pre-wrap;font-family:monospace;line-height:1.6\">{body}</div>"

    return ("<!doctype html><meta charset=utf-8><title>Span weights</title>"
            + line("SRA attention", row["attention"]) + line("ALERD omega", row["omega"]))


def build_report(dump, spans, examples, seed=0):
    rows = {}
    for row in dump:  # the same example recurs every epoch; the teacher is frozen, so the first copy is enough
        rows.setdefault(row["id"], row)
    rows = [r for r in rows.values() if "omega" in r and r["id"] in spans]
    if not rows:
        raise ValueError("the dump has no examples with omega; train with a non-attention span_weight_source")
    corr = correlation_summary(rows, seed)
    att, omg = (concentration(rows, spans, examples, k) for k in ("attention", "omega"))
    typed = [(span_types(examples[r["id"]], spans[r["id"]]), r) for r in rows]
    profiles = {"SRA attention": type_profile([(t, r["attention"]) for t, r in typed]),
                "ALERD omega": type_profile([(t, r["omega"]) for t, r in typed])}
    fmt = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    return "\n".join([
        "# ALERD omega versus SRA attention", "",
        f"Examples: {corr['n']} (with at least 4 spans and non-constant weights).", "",
        "## Within-example Spearman correlation", "",
        f"- median {corr['median']:.3f}, mean {corr['mean']:.3f}",
        f"- two independent random rankings: median {corr['random_median']:.3f}",
        f"- examples outside the central 95% range of two random rankings: {100 * corr['outside_95']:.1f}% (5% expected)",
        "",
        "## Where the weight goes (mean share of an example's weight)", "",
        "| Weight | First span | Top 10% of spans | Step stating the answer |", "|---|---|---|---|",
        f"| SRA attention | {fmt(att['first_span'])} | {fmt(att['top10'])} | {fmt(att['answer_step'])} |",
        f"| ALERD omega | {fmt(omg['first_span'])} | {fmt(omg['top10'])} | {fmt(omg['answer_step'])} |", "",
        "## Weight by span type (mean of N x weight over the spans of the type; 1.00 = uniform)", "",
        *type_table(profiles), ""]), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--html", help="write one example coloured by both weights")
    parser.add_argument("--example-id", type=int, help="example for the HTML (default: the median-length one)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]  # noqa: E731
    spans = {r["id"]: r for r in load(args.spans)}
    examples = {r["id"]: r for r in load(args.data)}
    report, rows = build_report(load(args.dump), spans, examples, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    if args.html:
        by_id = {r["id"]: r for r in rows}
        if args.example_id is not None:
            chosen = by_id[args.example_id]
        else:
            chosen = sorted(rows, key=lambda r: len(r["omega"]))[len(rows) // 2]
        Path(args.html).write_text(example_html(examples[chosen["id"]], spans[chosen["id"]], chosen), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
