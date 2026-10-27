r"""Build the SRA training jsonl from GSM8K.

    python prepare_sra_data.py clean --output work/gsm8k-clean.jsonl
    python prepare_sra_data.py clean-s1k --output work/s1k-clean.jsonl        # s1K-1.1 instead of GSM8K
    python prepare_sra_data.py build --clean work/gsm8k-clean.jsonl --student-tokenizer <student> \
        --teacher-tokenizer <teacher> --output work/<student>/train-sra.jsonl

`clean` reads openai/gsm8k (config main, split train; an HF id, a snapshot directory or a local jsonl) and writes one row
per problem: the solution without the calculator annotations <<...>>, the `#### N` line replaced by the sentence
`The final answer is \boxed{N}.` (thousands separators dropped from N), and `answer` = N, the target of the answer gain.
Every line of the solution is a step; `step_char_ends` are the character ends of the lines inside `output`.

`clean-s1k` reads simplescaling/s1K-1.1 and keeps, for every problem whose `deepseek_attempt` ends in a `\boxed{}`
answer, the attempt itself (the final write-up after the long thinking, which is not used) as `output` and its last
boxed answer as `answer`. A step is a paragraph: the text up to and including the line breaks that follow it.

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
from prompts import build_prompt, wrap_user

STEP = re.compile(r"[^\n]+\n*")
MATH_COT_PREFIX = "Solve the following math problem step-by-step."
TSDKD_DATASET = "Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen"
TSDKD_FILE = "Qwen2.5-14B-Instruct_train_all.jsonl"
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


def convert_s1k_row(row):
    """{question, deepseek_attempt} of s1K-1.1 -> {question, output, answer, step_char_ends}."""
    output = row["deepseek_attempt"].strip()
    answer = last_boxed(output)
    if not answer:
        raise ValueError("the attempt has no final boxed answer")
    return {"question": row["question"].strip(), "output": output, "answer": answer,
            "step_char_ends": [m.end() for m in STEP.finditer(output)]}


def convert_tsdkd_row(row):
    """TSD-KD release row {prompt: [user turn], response} -> {user_prompt, question, output, answer, step_char_ends}.

    The release's own user turn is the prompt (no extra instruction); the teacher's response is the training output.
    """
    user_prompt = row["prompt"][0]["content"].strip()
    output = row["response"].strip()
    answer = last_boxed(output)
    if not answer:
        raise ValueError("the response has no final boxed answer")
    return {"user_prompt": user_prompt, "question": user_prompt, "output": output, "answer": answer,
            "step_char_ends": [m.end() for m in STEP.finditer(output)]}


def clean_tsdkd_rows(rows):
    """Keep the Math CoT rows (by instruction prefix) that end in a boxed answer; ids are the original row numbers."""
    kept, counts = [], {"rows": 0, "math_cot": 0, "without_boxed": 0}
    for index, row in enumerate(rows):
        counts["rows"] += 1
        if not row["prompt"][0]["content"].lstrip().startswith(MATH_COT_PREFIX):
            continue
        counts["math_cot"] += 1
        try:
            converted = convert_tsdkd_row(row)
        except ValueError:
            counts["without_boxed"] += 1
            continue
        converted["id"] = index
        kept.append(converted)
    steps = [len(r["step_char_ends"]) for r in kept]
    return kept, {**counts, "kept": len(kept), "steps_median": statistics.median(steps), "steps_max": max(steps)}


def clean_s1k_rows(rows):
    kept, skipped = [], 0
    for index, row in enumerate(rows):
        try:
            converted = convert_s1k_row(row)
        except ValueError:
            skipped += 1
            continue
        converted["id"] = index
        kept.append(converted)
    steps = [len(r["step_char_ends"]) for r in kept]
    return kept, {"rows": len(kept), "skipped_without_boxed": skipped, "steps_median": statistics.median(steps),
                  "steps_max": max(steps)}


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


def build_rows(clean, student_tokenizer, teacher_tokenizer, max_len=None, teacher_max_len=None):
    """Rows with both prompts; with max_len / teacher_max_len, rows longer than that many tokens are dropped."""
    rows = []
    for row in clean:
        if "user_prompt" in row:  # the dataset brings its own instruction: no extra wrapper text
            prompt = wrap_user(student_tokenizer, row["user_prompt"])
            teacher_prompt = wrap_user(teacher_tokenizer, row["user_prompt"])
        else:
            prompt = build_prompt(student_tokenizer, row["question"])
            teacher_prompt = build_prompt(teacher_tokenizer, row["question"])
        check_boundary(student_tokenizer, prompt, row["output"], "student")
        check_boundary(teacher_tokenizer, teacher_prompt, row["output"], "teacher")
        if max_len is not None and len(student_tokenizer(prompt + row["output"], add_special_tokens=False
                                                         )["input_ids"]) > max_len:
            continue
        if teacher_max_len is not None and len(teacher_tokenizer(
                teacher_prompt + row["output"], add_special_tokens=False)["input_ids"]) > teacher_max_len:
            continue
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
    s1k = sub.add_parser("clean-s1k")
    s1k.add_argument("--dataset", default="simplescaling/s1K-1.1", help="HF id, snapshot dir or local jsonl")
    s1k.add_argument("--split", default="train")
    s1k.add_argument("--output", required=True)
    tsd = sub.add_parser("clean-tsdkd")
    tsd.add_argument("--dataset", default=TSDKD_DATASET, help="HF dataset id or a local jsonl of the release")
    tsd.add_argument("--output", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clean", required=True, help="output of the clean step")
    b.add_argument("--student-tokenizer", required=True)
    b.add_argument("--teacher-tokenizer", required=True)
    b.add_argument("--max-len", type=int, help="drop rows longer than this many student tokens (prompt + output)")
    b.add_argument("--teacher-max-len", type=int, help="drop rows longer than this many teacher tokens")
    b.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "clean-tsdkd":
        path = args.dataset
        if not Path(path).is_file():
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(args.dataset, TSDKD_FILE, repo_type="dataset")
        kept, stats = clean_tsdkd_rows(read_jsonl(path))
        write_jsonl(args.output, kept)
        print(json.dumps(stats))
        return

    if args.command == "clean-s1k":
        kept, stats = clean_s1k_rows(load_rows(args.dataset, args.split, None))
        write_jsonl(args.output, kept)
        print(json.dumps(stats))
        return

    if args.command == "clean":
        kept, stats = clean_rows(load_rows(args.dataset, args.split, args.config))
        write_jsonl(args.output, kept)
        print(json.dumps(stats))
        return

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    clean = read_jsonl(args.clean)
    rows = build_rows(clean, student, teacher, args.max_len, args.teacher_max_len)
    write_jsonl(args.output, rows)
    print(f"{len(rows)} samples -> {args.output} ({len(clean) - len(rows)} dropped as longer than the token limits)")
    print(length_stats(student, rows, "student"))
    print(length_stats(teacher, rows, "teacher"))


if __name__ == "__main__":
    main()
