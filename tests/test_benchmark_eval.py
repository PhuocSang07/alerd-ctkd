import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_eval  # noqa: E402
from answer_scoring import extract_boxed, grade_math_verify  # noqa: E402
from collect_results import load_scores  # noqa: E402
from helpers import CHAT_TEMPLATE, make_text, train_tokenizer  # noqa: E402
from benchmark_eval import gsm8k_records, main, score_file  # noqa: E402

ROWS = [{"question": f"Tom has {n} apples and buys 3 more. How many now?", "answer": f"3 more\n#### {n + 3},000"}
        for n in range(4)]


def test_extract_boxed_takes_the_last_answer_with_nested_braces():
    assert extract_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert extract_boxed("no answer here") is None


def test_grading_without_math_verify_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(ImportError, match="math-verify"):
        grade_math_verify(["x"], "1")


def test_score_file_reports_pass_at_one_and_pass_at_k(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_eval, "label_generations",
                        lambda texts, gold: [int(f"boxed{{{gold}}}" in t) for t in texts])
    rows = [{"id": 0, "gold": "2", "task_type": "math", "generations": [
                {"text": "\\boxed{2}", "finish_reason": "stop", "n_tokens": 4},
                {"text": "\\boxed{3}", "finish_reason": "length", "n_tokens": 6}]},
            {"id": 1, "gold": "5", "task_type": "math", "generations": [
                {"text": "no", "finish_reason": "stop", "n_tokens": 2},
                {"text": "also no", "finish_reason": "stop", "n_tokens": 2}]}]
    path = tmp_path / "gsm8k.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    summary = score_file(path)
    assert summary["pass@1"] == 0.25 and summary["pass@2"] == 0.5 and summary["n_problems"] == 2
    assert summary["truncation_rate"] == 0.25 and summary["no_boxed_answer_rate"] == 0.5 and summary["length"] == 3.5


@pytest.mark.parametrize("chat", [False, True])
def test_main_builds_training_prompts_and_writes_the_summary_the_tables_read(tmp_path, monkeypatch, chat):
    rng = random.Random(0)
    tokenizer = train_tokenizer("bpe", [make_text(rng, 6) for _ in range(100)])
    if chat:
        tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(tmp_path / "base")

    math = [{"id": 0, "question": "1+1?", "prompt": "unused", "gold": "2", "task_type": "math"}]
    monkeypatch.setitem(benchmark_eval.BENCHMARKS, "gsm8k", lambda: gsm8k_records(ROWS))
    monkeypatch.setitem(benchmark_eval.BENCHMARKS, "math500", lambda: math)
    seen = []
    real_generate_prompts = benchmark_eval.build_prompts

    def fake_generate(model_path, records, config):
        prompts = real_generate_prompts(model_path, records, config)
        seen.append((records[0]["gold"], prompts[0]["prompt_token_ids"], config["max_model_len"]))
        yield records, [[{"text": f"so \\boxed{{{r['gold']}}}", "finish_reason": "stop", "n_tokens": 4}] * 3
                        for r in records]

    monkeypatch.setattr(benchmark_eval, "generate", fake_generate)
    monkeypatch.setattr(benchmark_eval, "label_generations",
                        lambda texts, gold: [int(f"boxed{{{gold}}}" in t) for t in texts])
    main(["--model", str(tmp_path / "ckpt"), "--tag", "t-s42", "--base-model", str(tmp_path / "base"),
          "--benchmarks", "gsm8k,math500", "--n-samples", "3", "--temperature", "0.6", "--top-p", "0.9",
          "--max-tokens", "64", "--max-model-len", "256", "--results-dir", str(tmp_path / "results")])

    assert seen[0][0] == "3000" and seen[0][2] == 256  # gold without the thousands separator
    expected_prefix = tokenizer.encode(("<|user|>\n" if chat else "") + "Please reason step by step",
                                       add_special_tokens=False)
    assert seen[0][1][:len(expected_prefix)] == expected_prefix
    summary = json.loads((tmp_path / "results" / "t-s42" / "summary.json").read_text())
    assert [row["benchmark"] for row in summary] == ["gsm8k", "math500"]
    assert all(row["pass@1"] == 1.0 and row["pass@3"] == 1.0 for row in summary)
    assert summary[0]["n_problems"] == 4 and summary[0]["samples_per_problem"] == 3
    scores = load_scores(tmp_path / "results", "t-s42")
    assert scores[0] == pytest.approx((100.0, 100.0)) and scores[2] == pytest.approx((100.0, 100.0))


def test_pass_at_3_is_estimated_without_bias_from_more_than_three_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_eval, "label_generations", lambda texts, gold: [extract_boxed(t) == gold for t in texts])
    right, wrong = "so \\boxed{7}", "so \\boxed{8}"
    rows = [{"id": 0, "gold": "7", "generations": [{"text": t, "finish_reason": "stop", "n_tokens": 3}
                                                   for t in (right, right, wrong, wrong, wrong)]},
            {"id": 1, "gold": "7", "generations": [{"text": wrong, "finish_reason": "stop", "n_tokens": 3}] * 5}]
    path = tmp_path / "aime24.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    summary = benchmark_eval.score_file(path)
    assert summary["pass@5"] == pytest.approx(0.5)
    assert summary["pass@3"] == pytest.approx((1 - 1 / 10 + 0.0) / 2)  # 1 - C(3,3)/C(5,3) for the first problem, 0 for the second
