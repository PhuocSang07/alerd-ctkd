r"""Token-level answer gain for any scoring model (student or teacher), and the file format the trainer reads.

    python token_gain.py records --data work/<student>/train-sra.jsonl --tokenizer <model> --side student|teacher \
        --output work/<student>/tokens-<model>.jsonl
    python answer_gain.py --data-path work/<student>/tokens-<model>.jsonl --model <model> \
        --output work/<student>/gain-raw-<model>.json
    python token_gain.py pack --records work/<student>/tokens-<model>.jsonl --gains work/<student>/gain-raw-<model>.json \
        --model <model> --tokenizer <model> --output work/<student>/gain-tokens-<model>.json

`records` writes the input of answer_gain.py with every response token as its own "step", so that script
measures b_t = log m(a* | x, y_{<=t}, probe) at every token boundary of the scoring model m (t = 0 included) and returns
the token gains g_t = b_t - b_{t-1}. The text is the model's own prompt + the shared output, tokenized jointly exactly as
the trainer tokenizes it. `line_steps` holds the token span of each solution line for the step-level gains and for
score_predictability.py.

`pack` turns those gains into gain-tokens-<model>.json:
    {"model", "tokenizer", "samples": {id: {"hash": sha1 of prompt + output, "start": first response token,
                                            "token_gains": [g_t], "step_gains": [sum of g_t over each line]}}}
A span covering response tokens [a, b) has gain b_b - b_a = sum(token_gains[a - start : b - start]); the gain of a step is
the sum over the tokens that end inside it, the same rule that assigns a span to a step.

    python token_gain.py signals --spans work/spans.jsonl --student-gains <student gain-tokens> \
        --teacher-gains <teacher gain-tokens> --output work/span-gains.json

`signals` exports the two signals of the span-weighted losses per span, u^S (the student's span gain, for the CE weights)
and u^T - u^S (teacher minus student, for the KL weights): {id: {"hash", "teacher_hash", "student", "teacher", "diff"}}.
"""
import argparse
import json
import os
from itertools import accumulate

from data_utils import text_hash
from span_index import step_of_char


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def token_record(tokenizer, row, side):
    prompt = row["prompt"] if side == "student" else row.get("teacher_prompt", row["prompt"])
    full = prompt + row["output"]
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not 0 < start < len(ids) or offsets[start - 1][1] > len(prompt) or offsets[start][0] < len(prompt):
        raise ValueError(f"id {row['id']}: a token straddles the prompt/output boundary")
    step_ends = [len(prompt) + e for e in row["step_char_ends"]]
    first = [None] * len(step_ends)
    last = [None] * len(step_ends)
    for t in range(start, len(ids)):
        k = step_of_char(step_ends, offsets[t][1] - 1)
        first[k] = t if first[k] is None else first[k]
        last[k] = t + 1
    if any(a is None for a in first):
        raise ValueError(f"id {row['id']}: a step holds no token")
    return {"id": row["id"], "prompt": prompt, "response": row["output"], "input_ids": ids,
            "response_token_span": [start, len(ids)],
            "steps": [{"token_start": t, "token_end": t + 1} for t in range(start, len(ids))],
            "line_steps": [{"token_start": a, "token_end": b} for a, b in zip(first, last)],
            "hash": text_hash(full)}


def pack_gains(records, raw_gains, model, tokenizer_name):
    samples = {}
    for record in records:
        gains = raw_gains.get(str(record["id"]))
        if gains is None:
            raise ValueError(f"no gains for id {record['id']} (a probe longer than the model's context is skipped)")
        start = record["response_token_span"][0]
        if len(gains) != len(record["steps"]):
            raise ValueError(f"id {record['id']}: {len(gains)} gains for {len(record['steps'])} tokens")
        prefix = [0.0] + list(accumulate(gains))
        step_gains = [prefix[s["token_end"] - start] - prefix[s["token_start"] - start] for s in record["line_steps"]]
        samples[str(record["id"])] = {"hash": record["hash"], "start": start, "token_gains": gains,
                                      "step_gains": step_gains}
    return {"model": model, "tokenizer": tokenizer_name, "samples": samples}


class GainFile:
    """gain-tokens-<model>.json with the consistency checks the trainer relies on."""

    def __init__(self, path, role):
        self.path, self.role = path, role
        if not os.path.exists(path):
            raise FileNotFoundError(f"{role} gain file {path} not found; score it first (scripts/score_gain.sh)")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"{path}: not a gain-tokens file (run token_gain.py pack)")
        self.model, self.tokenizer, self.samples = payload.get("model"), payload.get("tokenizer"), payload["samples"]

    def entry(self, rid, expected_hash, expected_start, last_token):
        """The sample's gains, after checking that they were scored on the text and tokens the spans refer to."""
        entry = self.samples.get(str(rid))
        if entry is None:
            raise ValueError(f"{self.path}: no {self.role} gains for id {rid}")
        if entry["hash"] != expected_hash:
            raise ValueError(f"{self.path}: id {rid} was scored on a different text than the span index "
                             f"({self.role} hash mismatch, tokenizer {self.tokenizer}); rebuild one of them")
        if entry["start"] != expected_start or expected_start + len(entry["token_gains"]) < last_token:
            raise ValueError(f"{self.path}: id {rid} has a different response start or length than the span index "
                             f"({self.role} tokens)")
        return entry


def span_gains(entry, token_ranges):
    """Gain of each span from the token gains: sum over its [start, end) token range (absolute indices)."""
    prefix = [0.0] + list(accumulate(entry["token_gains"]))
    offset = entry["start"]
    return [prefix[b - offset] - prefix[a - offset] for a, b in token_ranges]


def span_signals(span_row, student_file, teacher_file):
    """Per-span gains of both scoring models for one indexed sample; each file checks the hash and token range."""
    rid = span_row["id"]
    gains = {}
    for role, file, hash_key in (("student", student_file, "hash"), ("teacher", teacher_file, "teacher_hash")):
        tokens = span_row[f"{role}_tokens"]
        entry = file.entry(rid, span_row[hash_key], tokens[0][0], tokens[-1][1])
        gains[role] = span_gains(entry, tokens)
    return {"hash": span_row["hash"], "teacher_hash": span_row["teacher_hash"], "student": gains["student"],
            "teacher": gains["teacher"], "diff": [t - s for t, s in zip(gains["teacher"], gains["student"])]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("records")
    r.add_argument("--data", required=True, help="train-sra.jsonl")
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--side", choices=("student", "teacher"), required=True,
                   help="which prompt of the row the scoring model sees")
    r.add_argument("--output", required=True)
    g = sub.add_parser("signals")
    g.add_argument("--spans", required=True, help="spans.jsonl from span_index.py")
    g.add_argument("--student-gains", required=True, help="gain-tokens json of the student (pack)")
    g.add_argument("--teacher-gains", required=True, help="gain-tokens json of the teacher (pack)")
    g.add_argument("--output", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--records", required=True)
    p.add_argument("--gains", required=True, help="output of answer_gain.py")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "records":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        records = [token_record(tokenizer, row, args.side) for row in read_jsonl(args.data)]
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"{len(records)} records, {sum(len(r['steps']) for r in records):,} token probes -> {args.output}")
        return

    if args.command == "signals":
        student, teacher = GainFile(args.student_gains, "student"), GainFile(args.teacher_gains, "teacher")
        signals = {str(row["id"]): span_signals(row, student, teacher) for row in read_jsonl(args.spans)}
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(signals, handle)
        print(f"wrote span gains of {len(signals)} samples -> {args.output}")
        return

    with open(args.gains, encoding="utf-8") as handle:
        raw = json.load(handle)
    packed = pack_gains(read_jsonl(args.records), raw, args.model, args.tokenizer)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(packed, handle)
    print(f"wrote {len(packed['samples'])} samples -> {args.output}")


if __name__ == "__main__":
    main()
