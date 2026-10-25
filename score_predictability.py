"""Mean token log-probability of each step under the frozen student (the "predictability" step score).

    python score_predictability.py --data work/<student>/tokens-<student>.jsonl --model <student> \\
        --output work/<student>/predictability.json

Reads the records of `token_gain.py records` and writes {id: [mean log p of the line's tokens given the prefix]},
one value per solution line (`line_steps`).
"""
import argparse
import json

import torch


def step_mean_logprobs(model, record):
    ids = torch.tensor([record["input_ids"]], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1).gather(1, ids[0, 1:, None])[:, 0]  # logp[t] scores token t + 1
    return [logp[step["token_start"] - 1: step["token_end"] - 1].mean().item() for step in record["line_steps"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None).eval()
    with open(args.data, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle][: args.limit]
    scores = {record["id"]: step_mean_logprobs(model, record) for record in records}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(scores, handle)
    print(f"wrote {len(scores)} samples -> {args.output}")


if __name__ == "__main__":
    main()
