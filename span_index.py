"""Offline span index: the spans the collator will build at train time, plus the step each one belongs to.

    python span_index.py --data work/train-sra.jsonl --student-tokenizer <s> --teacher-tokenizer <t> \\
        --max-len 4096 --output work/spans.jsonl

One row per sample: id, hash, char_ends (student character end of each span, in prompt + output),
student_tokens / teacher_tokens ([start, end) per span), step_of_span (step holding the span's last
character, None without step_char_ends) and crosses_step (the span's first and last character fall in
different steps). The span computation is sample_span_bounds from data_utils, the function the collator uses.
"""
import argparse
import hashlib
import json
import statistics
from bisect import bisect_right

from data_utils import LLMDataset, sample_span_bounds


def sample_spans(example, student_tokenizer, teacher_tokenizer, max_len, max_spans=None):
    """Spans of one example tokenized on its own (no batch padding), as the collator tokenizes it."""
    full = example["prompt"] + example["output"]
    student = student_tokenizer(full, truncation=True, max_length=max_len, return_offsets_mapping=True)
    teacher = teacher_tokenizer(full, truncation=True, max_length=max_len, return_offsets_mapping=True)
    bounds = sample_span_bounds(student["offset_mapping"], teacher["offset_mapping"],
                                example["s_prompt_len"], example["t_prompt_len"], max_spans)
    return bounds, student["offset_mapping"]


def step_of_char(step_ends, char):
    """Index of the step whose character range [previous end, end) holds `char` (clamped to the last step)."""
    return min(bisect_right(step_ends, char), len(step_ends) - 1)


def span_row(example, bounds, student_offsets):
    prompt_chars = len(example["prompt"])
    char_ends = [int(student_offsets[bounds[n + 1][0] - 1][1]) for n in range(len(bounds) - 1)]
    row = {
        "id": example.get("id"),
        "hash": hashlib.sha1((example["prompt"] + example["output"]).encode("utf-8")).hexdigest(),
        "char_ends": char_ends,
        "student_tokens": [[bounds[n][0], bounds[n + 1][0]] for n in range(len(bounds) - 1)],
        "teacher_tokens": [[bounds[n][1], bounds[n + 1][1]] for n in range(len(bounds) - 1)],
        "step_of_span": None,
        "crosses_step": None,
    }
    if example.get("step_char_ends"):
        ends = [prompt_chars + e for e in example["step_char_ends"]]
        starts = [prompt_chars] + char_ends[:-1]
        last = [step_of_char(ends, c - 1) for c in char_ends]
        first = [step_of_char(ends, c) for c in starts]
        row["step_of_span"] = last
        row["crosses_step"] = [a != b for a, b in zip(first, last)]
    return row


def build_rows(dataset, student_tokenizer, teacher_tokenizer, max_len, max_spans=None):
    rows = []
    for example in dataset.dataset:
        bounds, offsets = sample_spans(example, student_tokenizer, teacher_tokenizer, max_len, max_spans)
        rows.append(span_row(example, bounds, offsets))
    return rows


def summarize(values):
    ordered = sorted(values)
    return (f"median={statistics.median(ordered):.0f} p95={ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]} "
            f"max={ordered[-1]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="SRA jsonl from prepare_sra_data.py convert")
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--max-spans", type=int, default=None, help="cap on spans per sample; default keeps all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    print(f"padding_side student={student.padding_side} teacher={teacher.padding_side}")
    dataset = LLMDataset(args.data, student, teacher, args.max_len // 2)
    rows = build_rows(dataset, student, teacher, args.max_len, args.max_spans)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{len(rows)} samples; spans/sample {summarize([len(r['char_ends']) for r in rows])}")
    if rows and rows[0]["step_of_span"] is not None:
        crossing = sum(sum(r["crosses_step"]) for r in rows)
        total = sum(len(r["char_ends"]) for r in rows)
        print(f"spans crossing a step boundary: {crossing}/{total}")


if __name__ == "__main__":
    main()
