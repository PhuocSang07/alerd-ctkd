r"""Generate with vLLM on GSM8K test and MATH500, grade with Math-Verify, write pass@1 / pass@k summaries.

    python benchmark_eval.py --model <checkpoint> --tag <tag> --base-model <student> \
        --benchmarks gsm8k,math500 --n-samples 3 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 ...

* `gsm8k`: openai/gsm8k, config main, split test (1,319 problems), gold = the number after `####` without thousands
  separators. `math500`: HuggingFaceH4/MATH-500, split test, gold = the `answer` column. Both use the same instruction.
* Prompts come from prompts.build_prompt: the instruction as a user turn of the model's chat template when its tokenizer
  has one, the raw instruction otherwise (GPT-2). That is the prompt of the training data, so train and eval agree for
  both kinds of model. A prompt that does not fit the context stops the run before vLLM starts.
* Raw generations are written to <results-dir>/<tag>/raw/<benchmark>.jsonl and a run always regenerates them;
  `--rescore <raw file>` grades an existing file without a GPU. The summary is <results-dir>/<tag>/summary.json.
"""
import argparse
import json
import os
from pathlib import Path

from answer_scoring import grade_math_verify
from prompts import INSTRUCTION, build_prompt, has_chat_template

# On an offline server, BENCH_DATA_ROOT/<local dir name> replaces the Hugging Face repo id.
BENCH_DATA_ROOT = os.environ.get("BENCH_DATA_ROOT")


def dataset_path(repo_id, local_dir_name):
    return f"{BENCH_DATA_ROOT}/{local_dir_name}" if BENCH_DATA_ROOT else repo_id


def gsm8k_gold(answer):
    return answer.rpartition("####")[2].strip().replace(",", "")


def math_records(rows, question_key, answer_key, gold=str):
    return [{"id": index, "question": row[question_key], "prompt": INSTRUCTION.format(question=row[question_key]),
             "gold": gold(row[answer_key]), "task_type": "math"} for index, row in enumerate(rows)]


def gsm8k_records(rows):
    return math_records(rows, "question", "answer", gsm8k_gold)


def load_gsm8k():
    from datasets import load_dataset

    return gsm8k_records(load_dataset(dataset_path("openai/gsm8k", "gsm8k"), "main", split="test"))


def load_math500():
    from datasets import load_dataset

    return math_records(load_dataset(dataset_path("HuggingFaceH4/MATH-500", "MATH-500"), split="test"),
                        "problem", "answer")


BENCHMARKS = {"gsm8k": load_gsm8k, "math500": load_math500}


def build_prompts(model_path, records, config):
    """Token ids of each record's prompt; the tokenizer of `base_model` (the student) decides the wrapping."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model") or model_path)
    prompts = [{"prompt_token_ids": tokenizer.encode(build_prompt(tokenizer, record["question"]),
                                                     add_special_tokens=False)} for record in records]
    longest = max(len(p["prompt_token_ids"]) for p in prompts)
    if longest >= config["max_model_len"]:
        raise ValueError(f"a prompt of {longest} tokens does not fit max_model_len {config['max_model_len']}")
    kind = "chat template" if has_chat_template(tokenizer) else "raw text"
    print(f"{len(prompts)} prompts as {kind}; longest {longest} tokens", flush=True)
    return prompts


def generate(model_path, records, config):
    """Yield (records_slice, generations) a batch at a time, so finished problems are written as they land."""
    from vllm import LLM, SamplingParams

    prompts = build_prompts(model_path, records, config)
    llm = LLM(model=model_path, max_model_len=config["max_model_len"],
              gpu_memory_utilization=config["gpu_memory_utilization"], dtype="bfloat16",
              enforce_eager=config["enforce_eager"], disable_log_stats=True)
    sampling = SamplingParams(n=config["n_samples"], temperature=config["temperature"], top_p=config["top_p"],
                              repetition_penalty=config["repetition_penalty"], max_tokens=config["max_tokens"],
                              seed=config["seed"])
    batch_size = config["batch_size"] or len(records)
    for start in range(0, len(records), batch_size):
        stop = start + batch_size
        outputs = llm.generate(prompts[start:stop], sampling, use_tqdm=False)
        yield records[start:stop], [
            [{"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)} for c in output.outputs]
            for output in outputs]


def label_generations(texts, gold):
    return grade_math_verify(texts, gold)


def score_file(path):
    """Pass@1 (mean over all samples) and pass@k (share of problems with any correct sample) of a raw file."""
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle]
    correct = total = truncated = unboxed = token_sum = solved = samples_per_problem = 0
    for row in rows:
        texts = [generation["text"] for generation in row["generations"]]
        labels = label_generations(texts, row["gold"])
        for text, generation in zip(texts, row["generations"]):
            total += 1
            truncated += int(generation.get("finish_reason") == "length")  # never finished its answer
            unboxed += int(len(text) > 0 and "\\boxed" not in text)
            token_sum += generation.get("n_tokens", 0)
        correct += sum(labels)
        solved += int(sum(labels) > 0)
        samples_per_problem = max(samples_per_problem, len(row["generations"]))
    summary = {
        "benchmark": Path(path).stem,
        "accuracy": correct / total if total else 0.0,
        "pass@1": correct / total if total else 0.0,
        "samples_per_problem": samples_per_problem,
        "length": token_sum / total if total else 0.0,
        "n_problems": len(rows),
        "n_generations": total,
        "truncation_rate": truncated / total if total else 0.0,
        "no_boxed_answer_rate": unboxed / total if total else 0.0,
    }
    if samples_per_problem > 1:
        summary[f"pass@{samples_per_problem}"] = solved / len(rows)
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="checkpoint path")
    parser.add_argument("--tag", help="short name used in output paths (default: directory name of --model)")
    parser.add_argument("--benchmarks", default="gsm8k,math500")
    parser.add_argument("--rescore", help="grade an existing raw generations file and exit")
    parser.add_argument("--base-model", help="supplies the tokenizer and prompt format (the student's)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, help="problems per generate() call (default: a whole benchmark)")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    return parser


def print_summary_table(tag, summaries):
    k = max(s["samples_per_problem"] for s in summaries)
    cols = ["benchmark", "pass@1"] + ([f"pass@{k}"] if k > 1 else []) + ["length", "truncated", "no_boxed", "n"]
    rows = [[s["benchmark"], f"{s['pass@1']:.2%}", *([f"{s[f'pass@{k}']:.2%}"] if k > 1 else []),
             f"{s['length']:.0f}", f"{s['truncation_rate']:.2%}", f"{s['no_boxed_answer_rate']:.2%}",
             str(s["n_problems"])] for s in summaries]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]

    def fmt(row):
        return "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))

    print(f"\n=== {tag} ===")
    print(fmt(cols))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rescore:
        print(json.dumps(score_file(args.rescore), indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --rescore is given")
    config = {"base_model": args.base_model, "temperature": args.temperature, "top_p": args.top_p,
              "repetition_penalty": args.repetition_penalty, "n_samples": args.n_samples,
              "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
              "gpu_memory_utilization": args.gpu_memory_utilization, "batch_size": args.batch_size,
              "enforce_eager": args.enforce_eager, "seed": args.seed}
    tag = args.tag or Path(args.model).name
    run_dir = Path(args.results_dir) / tag
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.benchmarks.split(","):
        raw_path = raw_dir / f"{name}.jsonl"
        records = BENCHMARKS[name]()
        print(f"[{name}] generating for {len(records)} problems x {args.n_samples}", flush=True)
        with raw_path.open("w") as handle:
            for batch_records, batch_completions in generate(args.model, records, config):
                for record, completions in zip(batch_records, batch_completions):
                    handle.write(json.dumps({"id": record["id"], "gold": record["gold"],
                                             "task_type": record["task_type"], "generations": completions}) + "\n")
        summary = score_file(raw_path)
        summary["model"] = tag
        summaries.append(summary)
        k = summary["samples_per_problem"]
        print(f"[{name}] pass@1 = {summary['pass@1']:.2%}  "
              + (f"pass@{k} = {summary[f'pass@{k}']:.2%}  " if k > 1 else "")
              + f"length = {summary['length']:.0f} tok  truncated = {summary['truncation_rate']:.2%}", flush=True)

    results_path = run_dir / "summary.json"
    results_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n{tag}: Overall pass@1 = {sum(s['pass@1'] for s in summaries) / len(summaries):.2%} -> {results_path}")
    print_summary_table(tag, summaries)


if __name__ == "__main__":
    main()
