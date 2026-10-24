r"""Build the SRA training jsonl from s1K-1.1 `deepseek_attempt` solutions.

Two stages around the `sgl` pipeline (sgl is only called, never modified; put its `src` on PYTHONPATH).
Run the middle step once per student, since the chat template and tokenizer differ (scripts/score_gain.sh does it):

    python prepare_sra_data.py filter  --output work/s1k-boxed.jsonl
    python -m sgl.data.prepare --dataset-name work/s1k-boxed.jsonl --tokenizer <student> --chat-template \
        --no-enable-thinking --palign-prompt --max-tokens 8192 --output-path work/train-segmented.jsonl
    python prepare_sra_data.py convert --segmented work/train-segmented.jsonl --student-tokenizer <student> \
        --output work/train-sra.jsonl

`filter` keeps every row whose `deepseek_attempt` is non-empty and has an extractable final \boxed{} (the gain
target), whether or not the solution is graded correct, and writes {question, instruction, output =
deepseek_attempt}, the shape sgl.data.prepare reads verbatim (no `<think>` wrapper or appended answer).
`convert` turns the segmented records into SRA rows {id, prompt, output, step_char_ends}; step_char_ends are
the character ends of each step inside `output`, taken from the student tokenizer's offset mapping.
"""
import argparse
import json
import statistics
from pathlib import Path


def load_rows(dataset, split):
    from datasets import load_dataset

    if Path(dataset).is_file():
        return load_dataset("json", data_files=dataset, split="train")
    if Path(dataset).is_dir():  # offline HF snapshot
        files = sorted(str(p) for p in Path(dataset).rglob("*.parquet"))
        return load_dataset("parquet", data_files=files, split="train")
    return load_dataset(dataset, split=split)


def filter_rows(rows):
    """Keep rows whose attempt ends in an extractable \boxed{} and shape them for sgl.data.prepare."""
    from sgl.data.s1k import last_boxed

    kept, empty_attempt, no_boxed, kept_grade_no = [], 0, 0, 0
    for row in rows:
        attempt = (row.get("deepseek_attempt") or "").strip()
        if not attempt:
            empty_attempt += 1
            continue
        if last_boxed(attempt) is None:
            no_boxed += 1
            continue
        kept_grade_no += str(row.get("deepseek_grade")).strip() == "No"
        kept.append({"question": row["question"].strip(), "instruction": "", "output": attempt,
                     "solution": (row.get("solution") or "").strip()})
    return kept, {"rows": len(rows), "kept": len(kept), "empty_attempt": empty_attempt,
                  "dropped_no_boxed": no_boxed, "kept_with_grade_no": kept_grade_no}


def step_char_ends(tokenizer, record):
    """Character end of each step within record['response'], from the student tokenizer's offsets."""
    start, end = record["response_token_span"]
    encoded = tokenizer(record["response"], add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != record["input_ids"][start:end]:
        raise ValueError(f"record {record['id']}: tokenizer does not reproduce the segmented token ids")
    offsets = encoded["offset_mapping"]
    return [offsets[step["token_end"] - start - 1][1] for step in record["steps"]]


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def convert_records(tokenizer, records):
    rows = []
    for record in records:
        rows.append({"id": record["id"], "prompt": record["prompt"], "output": record["response"],
                     "step_char_ends": step_char_ends(tokenizer, record)})
    return rows


def length_stats(tokenizer, rows):
    lengths = [len(tokenizer(row["prompt"] + row["output"])["input_ids"]) for row in rows]
    steps = [len(row["step_char_ends"]) for row in rows]
    return (f"kept {len(rows)} samples; tokens median={statistics.median(lengths):.0f} "
            f"p95={pct(lengths, 0.95)} max={max(lengths)}; steps median={statistics.median(steps):.0f} "
            f"p95={pct(steps, 0.95)} max={max(steps)}")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--dataset", default="simplescaling/s1K-1.1", help="HF id, snapshot dir or local jsonl")
    f.add_argument("--split", default="train")
    f.add_argument("--output", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--segmented", required=True, help="train-segmented.jsonl from sgl.data.prepare")
    c.add_argument("--student-tokenizer", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "filter":
        kept, stats = filter_rows(load_rows(args.dataset, args.split))
        write_jsonl(args.output, kept)
        print(json.dumps(stats))
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    records = [json.loads(line) for line in open(args.segmented, encoding="utf-8")]
    rows = convert_records(tokenizer, records)
    write_jsonl(args.output, rows)
    print(f"padding_side student={tokenizer.padding_side}")
    print(length_stats(tokenizer, rows))


if __name__ == "__main__":
    main()
