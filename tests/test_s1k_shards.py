import json
import os
import random
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_text, train_tokenizer  # noqa: E402
from prepare_sra_data import build_rows, clean_s1k_rows, convert_s1k_row  # noqa: E402
from token_gain import main as token_gain_main, merge_gain_files  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which("bash")

ATTEMPT = ("We compute the sum.\n\n1. **First step**:\n   - Add 2 and 3 to get 5.\n\n2. **Second step**:\n"
           "   - Double it to get 10.\n\nThus the answer is $\\boxed{10}$.\n")


def test_s1k_row_keeps_the_attempt_and_its_last_boxed_answer():
    row = convert_s1k_row({"question": " What is 2(2+3)? ", "deepseek_attempt": ATTEMPT})
    assert row["output"] == ATTEMPT.strip() and row["question"] == "What is 2(2+3)?"
    assert row["answer"] == "10"
    ends = row["step_char_ends"]
    assert ends[-1] == len(row["output"]) and ends == sorted(set(ends))
    # a step is a line with the line breaks that follow it: no step is empty or starts with a break
    starts = [0] + ends[:-1]
    assert all(row["output"][a:b].strip() and row["output"][a] != "\n" for a, b in zip(starts, ends))


def test_s1k_rows_without_a_boxed_answer_are_skipped_and_ids_are_the_original_rows():
    rows = [{"question": "a", "deepseek_attempt": ATTEMPT}, {"question": "b", "deepseek_attempt": "no answer here"},
            {"question": "c", "deepseek_attempt": ATTEMPT}]
    kept, stats = clean_s1k_rows(rows)
    assert [r["id"] for r in kept] == [0, 2]
    assert stats["rows"] == 2 and stats["skipped_without_boxed"] == 1


def test_build_drops_rows_longer_than_the_token_limits():
    rng = random.Random(0)
    tokenizer = train_tokenizer("bpe", [make_text(rng, 6) for _ in range(100)])
    short = {"id": 0, "question": "Find x.", "answer": "7", "output": "So x is \\boxed{7}.", "step_char_ends": [18]}
    long = dict(short, id=1, output=make_text(rng, 30) + " \\boxed{7}")
    long["step_char_ends"] = [len(long["output"])]
    [row] = build_rows([short], tokenizer, tokenizer)
    limit = len(tokenizer(row["prompt"] + row["output"], add_special_tokens=False)["input_ids"]) + 5
    kept = build_rows([short, long], tokenizer, tokenizer, max_len=limit)
    assert [r["id"] for r in kept] == [0]
    assert [r["id"] for r in build_rows([short, long], tokenizer, tokenizer)] == [0, 1]


def gain_payload(model, ids):
    return {"model": model, "tokenizer": model, "samples": {str(i): {"hash": f"h{i}", "start": 5, "token_gains": [0.1],
                                                                       "step_gains": [0.1]} for i in ids}}


def test_merged_shards_hold_every_sample_once_and_refuse_overlaps_or_mixed_models():
    merged = merge_gain_files([gain_payload("m", [0, 3]), gain_payload("m", [1, 4]), gain_payload("m", [2])])
    assert sorted(merged["samples"], key=int) == ["0", "1", "2", "3", "4"] and merged["model"] == "m"
    with pytest.raises(ValueError, match="more than one shard"):
        merge_gain_files([gain_payload("m", [0, 1]), gain_payload("m", [1])])
    with pytest.raises(ValueError, match="different models"):
        merge_gain_files([gain_payload("m", [0]), gain_payload("other", [1])])


def test_merge_command_writes_one_file(tmp_path):
    for i, ids in enumerate(([0, 2], [1])):
        (tmp_path / f"s{i}.json").write_text(json.dumps(gain_payload("m", ids)))
    token_gain_main(["merge", "--inputs", str(tmp_path / "s0.json"), str(tmp_path / "s1.json"),
                     "--output", str(tmp_path / "all.json")])
    assert set(json.loads((tmp_path / "all.json").read_text())["samples"]) == {"0", "1", "2"}


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_score_gain_shard_of_s1k_scores_its_rows_and_leaves_the_index_to_the_merge(tmp_path):
    shard = tmp_path / "work-s1k-shard1of3"
    (shard / "Qwen2.5-1.5B-Instruct").mkdir(parents=True)
    (shard / "Qwen2.5-1.5B-Instruct" / "train-sra.jsonl").write_text("".join(f'{{"id": {i}}}\n' for i in range(7)))
    env = dict(os.environ, PYTHON="echo", STUDENTS="Qwen/Qwen2.5-1.5B-Instruct", TEACHER="Qwen/Qwen2.5-14B-Instruct",
               SPAN_MODE="word", DATASET="s1k", SHARD="1/3")
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path / "work-s1k")], cwd=ROOT, env=env, check=True,
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert any("clean-s1k" in line and "s1k-clean.jsonl" in line for line in lines)
    assert any("prepare_sra_data.py build" in line and "--max-len 1024" in line and "--teacher-max-len 2560" in line
               for line in lines)
    assert sum("answer_gain.py" in line for line in lines) == 2  # the student and the teacher, on this shard
    assert not any("score_predictability.py" in line or "span_index.py" in line or "signals" in line for line in lines)
    # echo writes nothing, so the pre-made training file shows the dealing: rows 1, 4 of 0..6 go to shard 1 of 3
    kept = (shard / "Qwen2.5-1.5B-Instruct" / "train-sra.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in kept] == [1, 4]


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("env", [{"SHARD": "3/3"}, {"SHARD": "a/b"}, {"DATASET": "alpaca"}])
def test_score_gain_rejects_bad_shard_and_dataset_settings(tmp_path, env):
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path)], cwd=ROOT,
                            env=dict(os.environ, PYTHON="echo", **env), capture_output=True)
    assert result.returncode == 2
