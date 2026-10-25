r"""Build the SRA training jsonl from GSM8K.

    python prepare_sra_data.py clean --output work/gsm8k-clean.jsonl
    python prepare_sra_data.py build --clean work/gsm8k-clean.jsonl --student-tokenizer <student> \
        --teacher-tokenizer <teacher> --output work/<student>/train-sra.jsonl

`clean` reads openai/gsm8k (config main, split train; an HF id, a snapshot directory or a local jsonl) and writes one row
per problem: the solution without the calculator annotations <<...>>, the `#### N` line replaced by the sentence
`The final answer is \boxed{N}.` (thousands separators dropped from N), and `answer` = N, the target of the answer gain.
Every line of the solution is a step; `step_char_ends` are the character ends of the lines inside `output`.

`build` wraps the question in each model's prompt (prompts.py): `prompt` for the student and `teacher_prompt` for the
teacher. The output text is the same for every model. Prompt and output must tokenize independently: if a token of the
student or teacher would straddle the prompt/output boundary, the row fails the build.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from allocation import last_boxed
from prompts import build_prompt

CALCULATOR = re.compile(r"<<[^<>]*>>")
NUMBER = re.compile(r"-?\d+(\.\d+)?")


def load_rows(dataset, split, config="main"):
    from datasets import load_dataset

    if Path(dataset).is_file():
        return load_dataset("json", data_files=dataset, split="train")
    if Path(dataset).is_dir():  # offline HF snapshot
        files = sorted(str(p) for p in Path(dataset).rglob("*.parquet") if split in p.name)
        return load_dataset("parquet", data_files=files, split="train")
    return load_dataset(dataset, config, split=split)


def convert_row(row):
    """{question, answer} of GSM8K -> {question, output, answer, step_char_ends}."""
    body, marker, final = row["answer"].rpartition("####")
    if not marker:
        raise ValueError(f"no '####' line in the solution: {row['answer'][-60:]!r}")
    number = final.strip().replace(",", "")
    if not number:
        raise ValueError("empty final answer")
    lines = [CALCULATOR.sub("", line).strip() for line in body.strip().split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("solution has no steps")
    lines.append(f"The final answer is \\boxed{{{number}}}.")
    output, ends = "", []
    for line in lines:
        output += line + "\n"
        ends.append(len(output))
    output = output[:-1]  # no trailing newline: the last step ends where the text ends
    ends[-1] = len(output)
    if last_boxed(output) != number:
        raise ValueError("final boxed answer not recoverable")
    return {"question": row["question"].strip(), "output": output, "answer": number, "step_char_ends": ends}


def clean_rows(rows):
    kept = []
    for index, row in enumerate(rows):
        converted = convert_row(row)
        converted["id"] = index
        kept.append(converted)
    non_numeric = sum(NUMBER.fullmatch(r["answer"]) is None for r in kept)
    steps = [len(r["step_char_ends"]) for r in kept]
    stats = {"rows": len(kept), "non_numeric_answers": non_numeric, "steps_median": statistics.median(steps),
             "steps_max": max(steps)}
    return kept, stats


def check_boundary(tokenizer, prompt, output, label):
    """The prompt's tokens must be a prefix of the tokens of prompt + output (no token across the boundary)."""
    prompt_len = len(prompt)
    offsets = tokenizer(prompt + output, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if n_prompt >= len(offsets) or offsets[n_prompt - 1][1] > prompt_len or offsets[n_prompt][0] < prompt_len:
        raise ValueError(f"a {label} token straddles the prompt/output boundary")


def build_rows(clean, student_tokenizer, teacher_tokenizer):
    rows = []
    for row in clean:
        prompt = build_prompt(student_tokenizer, row["question"])
        teacher_prompt = build_prompt(teacher_tokenizer, row["question"])
        check_boundary(student_tokenizer, prompt, row["output"], "student")
        check_boundary(teacher_tokenizer, teacher_prompt, row["output"], "teacher")
        rows.append({"id": row["id"], "question": row["question"], "answer": row["answer"], "prompt": prompt,
                     "teacher_prompt": teacher_prompt, "output": row["output"],
                     "step_char_ends": row["step_char_ends"]})
    return rows


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def length_stats(tokenizer, rows, side):
    key = "prompt" if side == "student" else "teacher_prompt"
    lengths = [len(tokenizer(row[key] + row["output"], add_special_tokens=False)["input_ids"]) for row in rows]
    return (f"{side}: tokens median={statistics.median(lengths):.0f} p95={pct(lengths, 0.95)} max={max(lengths)}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("clean")
    c.add_argument("--dataset", default="openai/gsm8k", help="HF id, snapshot dir or local jsonl")
    c.add_argument("--config", default="main")
    c.add_argument("--split", default="train")
    c.add_argument("--output", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clean", required=True, help="output of the clean step")
    b.add_argument("--student-tokenizer", required=True)
    b.add_argument("--teacher-tokenizer", required=True)
    b.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "clean":
        kept, stats = clean_rows(load_rows(args.dataset, args.split, args.config))
        write_jsonl(args.output, kept)
        print(json.dumps(stats))
        return

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    rows = build_rows(read_jsonl(args.clean), student, teacher)
    write_jsonl(args.output, rows)
    print(f"{len(rows)} samples -> {args.output}")
    print(length_stats(student, rows, "student"))
    print(length_stats(teacher, rows, "teacher"))


if __name__ == "__main__":
    main()
