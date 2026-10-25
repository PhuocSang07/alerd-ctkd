r"""sgl.eval.evaluate with GSM8K test added and prompts built the way training builds them.

    PYTHONPATH=$SGL_SRC python run_sgl_eval.py --model <checkpoint> --tag <tag> --base-model <student> \
        --benchmarks gsm8k,math500 --n-samples 3 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 ...

All arguments are those of sgl.eval.evaluate (the sgl repository is only imported, never changed). Two things differ:

* `gsm8k` is a benchmark: openai/gsm8k, config main, split test (1,319 problems), gold = the number after `####`
  without thousands separators, prompt = the same instruction as the other benchmarks.
* Prompts come from prompts.build_prompt: the instruction as a user turn of the model's chat template when its tokenizer
  has one, the raw instruction otherwise (GPT-2). That is the prompt of the training data, so train and eval agree for
  both kinds of model, and the chat-template flags of sgl are not needed. A prompt that does not fit the context stops
  the run before vLLM starts.
"""
import sys

from prompts import INSTRUCTION, build_prompt, has_chat_template


def gsm8k_gold(answer):
    return answer.rpartition("####")[2].strip().replace(",", "")


def gsm8k_records(rows):
    return [{"id": index, "question": row["question"], "prompt": INSTRUCTION.format(question=row["question"]),
             "gold": gsm8k_gold(row["answer"]), "task_type": "math"} for index, row in enumerate(rows)]


def load_gsm8k():
    from datasets import load_dataset
    from sgl.eval.benchmarks import _dataset_path

    return gsm8k_records(load_dataset(_dataset_path("openai/gsm8k", "gsm8k"), "main", split="test"))


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


def main(argv=None):
    from sgl.eval import benchmarks, evaluate

    benchmarks.BENCHMARKS["gsm8k"] = load_gsm8k
    evaluate.build_prompts = build_prompts
    evaluate.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
