import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import CHAT_TEMPLATE, make_text, train_tokenizer  # noqa: E402
from collect_results import load_scores  # noqa: E402
from run_sgl_eval import gsm8k_records, main  # noqa: E402

ROWS = [{"question": f"Tom has {n} apples and buys 3 more. How many now?", "answer": f"3 more\n#### {n + 3},000"}
        for n in range(4)]


@pytest.fixture
def sgl():
    src = os.environ.get("SGL_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set SGL_SRC to run the sgl wrapper")
    sys.path.insert(0, src)
    try:
        from sgl.eval import benchmarks, evaluate
    except ImportError as error:
        pytest.skip(f"sgl is not importable here: {error}")
    finally:
        sys.path.remove(src)
    yield benchmarks, evaluate
    benchmarks.BENCHMARKS.pop("gsm8k", None)


@pytest.mark.parametrize("chat", [False, True])
def test_wrapper_adds_gsm8k_builds_training_prompts_and_writes_the_summary_the_tables_read(sgl, tmp_path, monkeypatch,
                                                                                         chat):
    benchmarks, evaluate = sgl
    rng = random.Random(0)
    tokenizer = train_tokenizer("bpe", [make_text(rng, 6) for _ in range(100)])
    if chat:
        tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(tmp_path / "base")

    math = [{"id": 0, "question": "1+1?", "prompt": "unused", "gold": "2", "task_type": "math"}]
    monkeypatch.setattr("run_sgl_eval.load_gsm8k", lambda: gsm8k_records(ROWS))
    monkeypatch.setitem(benchmarks.BENCHMARKS, "math500", lambda: math)
    seen = []

    def fake_generate(model_path, records, config):
        prompts = evaluate.build_prompts(model_path, records, config)  # the wrapper's builder, not sgl's
        seen.append((records[0]["gold"], prompts[0]["prompt_token_ids"], config["max_model_len"]))
        yield records, [[{"text": f"so \\boxed{{{r['gold']}}}", "finish_reason": "stop", "n_tokens": 4}] * 3
                        for r in records]

    monkeypatch.setattr(evaluate, "generate", fake_generate)
    monkeypatch.setattr(evaluate, "label_generations",
                        lambda texts, gold, task_type, grader: [int(f"boxed{{{gold}}}" in t) for t in texts])
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
