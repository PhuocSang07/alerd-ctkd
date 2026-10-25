"""Answer-information gain of a frozen model, one value per "step" of a response (needs a GPU with vLLM).

    gain_k = log p(answer | prompt, steps_1..k) - log p(answer | prompt, steps_1..k-1)

p(answer | context) is read by appending a fixed probe ("... the final answer is \\boxed{") and scoring the gold answer
tokens plus "}" with vLLM prompt_logprobs. All probes of a sample share the response prefix, so vLLM's prefix cache
makes this about one forward pass per sample. The gold answer is the last \\boxed{...} of the response.

    python answer_gain.py --data-path <records.jsonl> --model <model> --output <gains.json>

Each input record has `id`, `response`, `input_ids`, `response_token_span` and `steps` (each with `token_end`), as written
by token_gain.py. Output: {id: [gain per step]} JSON.
"""
import argparse
import json
import time
from pathlib import Path

from answer_scoring import extract_boxed

PROBE = "\n\nTherefore, the final answer is \\boxed{"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="records jsonl written by token_gain.py records")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data_path) as handle:
        records = [json.loads(line) for line in handle][:args.limit]
    probe_ids = tokenizer(PROBE, add_special_tokens=False)["input_ids"]

    prompts, index = [], []  # index[i] = (record id, step k (0 = before any step), n answer tokens)
    skipped = 0
    for record in records:
        answer = extract_boxed(record["response"])
        if not answer:
            skipped += 1
            continue
        answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
        cuts = [record["response_token_span"][0]] + [step["token_end"] for step in record["steps"]]
        for k, cut in enumerate(cuts):
            ids = record["input_ids"][:cut] + probe_ids + answer_ids
            if len(ids) >= args.max_model_len:
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            index.append((record["id"], k, len(answer_ids)))
    print(f"{len(prompts):,} probes over {len(records) - skipped} samples (skipped {skipped} without \\boxed)")

    started = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=True,
              max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs)
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
    print(f"scored {len(prompts):,} probes with {args.model} in {time.time() - started:.0f} s (model load included)")

    logp = {}  # (id, k) -> log p(answer | prefix up to step k)
    for (rid, k, n_answer), out, prompt in zip(index, outputs, prompts):
        ids = prompt["prompt_token_ids"]
        logp[(rid, k)] = sum(out.prompt_logprobs[position][ids[position]].logprob
                             for position in range(len(ids) - n_answer, len(ids)))

    gains = {}
    for record in records:
        rid, n_steps = record["id"], len(record["steps"])
        if (rid, 0) not in logp:
            continue
        values = [logp.get((rid, k)) for k in range(n_steps + 1)]
        for k in range(1, len(values)):  # a probe skipped for length inherits the last known value: zero gain
            if values[k] is None:
                values[k] = values[k - 1]
        gains[rid] = [values[k] - values[k - 1] for k in range(1, n_steps + 1)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(gains))
    print(f"wrote gains for {len(gains)} samples -> {args.output}")


if __name__ == "__main__":
    main()
