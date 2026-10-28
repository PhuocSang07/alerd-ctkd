import os
import random
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_text, train_tokenizer  # noqa: E402
from prepare_sra_data import MATH_COT_PREFIX, build_rows, clean_tsdkd_rows, convert_tsdkd_row  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which("bash")

RESPONSE = "First, add.\n\n1. **Step**: 2 + 3 = 5.\n\nSo the answer is $\\boxed{5}$."


def release_row(instruction, response):
    return {"instruction": instruction, "prompt": [{"role": "user", "content": instruction}], "response": response}


def test_row_keeps_the_release_prompt_and_the_teacher_response():
    instruction = MATH_COT_PREFIX + "\nPresent your final answer as \\boxed{Your Answer}.\nWhat is 2 + 3?"
    row = convert_tsdkd_row(release_row(instruction, " " + RESPONSE + "\n"))
    assert row["user_prompt"] == instruction and row["output"] == RESPONSE and row["answer"] == "5"
    assert row["step_char_ends"][-1] == len(RESPONSE)


def test_only_math_cot_rows_with_a_boxed_answer_are_kept_and_ids_are_the_release_rows():
    rows = [release_row(MATH_COT_PREFIX + "\nQ1", RESPONSE),
            release_row("Write Python code to solve the task.\nQ2", RESPONSE),
            release_row("Tool available:\n[1] Python\n" + MATH_COT_PREFIX + "\nQ3", RESPONSE),
            release_row(MATH_COT_PREFIX + "\nQ4", "no boxed answer"),
            release_row(MATH_COT_PREFIX + "\nQ5", RESPONSE)]
    kept, stats = clean_tsdkd_rows(rows)
    assert [r["id"] for r in kept] == [0, 4]
    assert (stats["rows"], stats["math_cot"], stats["without_boxed"], stats["kept"]) == (5, 3, 1, 2)


def test_prompt_is_the_release_user_turn_without_the_extra_instruction():
    tokenizer = train_tokenizer("bpe", [make_text(random.Random(0), 6) for _ in range(100)])
    row = convert_tsdkd_row(release_row(MATH_COT_PREFIX + "\nWhat is 2 + 3?", RESPONSE))
    row["id"] = 0
    [built] = build_rows([row], tokenizer, tokenizer)
    assert built["prompt"] == MATH_COT_PREFIX + "\nWhat is 2 + 3?" and "Please reason step by step" not in built["prompt"]


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_score_gain_cleans_the_release_and_uses_one_limit_for_both_models(tmp_path):
    env = dict(os.environ, PYTHON="echo", STUDENTS="Qwen/Qwen2.5-1.5B-Instruct", TEACHER="Qwen/Qwen2.5-14B-Instruct",
               DATASET="tsdkd", TSDKD_SOURCE="release.jsonl", LIMIT="500")
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path / "work-tsdkd")], cwd=ROOT, env=env, check=True,
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert any("clean-tsdkd" in line and "--dataset release.jsonl" in line and "tsdkd-clean.jsonl" in line for line in lines)
    assert any("prepare_sra_data.py build" in line and "--max-len 1024" in line and "--teacher-max-len 1024" in line
               and "--num-samples 500" in line and "--sample-seed" not in line for line in lines)
    assert not (tmp_path / "work-tsdkd-limit500" / "clean.tmp").exists()  # no head -n: the first K are taken after the length filter


def test_filtered_rows_pass_through_and_raw_release_rows_are_filtered():
    from prepare_sra_data import clean_tsdkd_source

    filtered = [{"id": 3, "output": RESPONSE, "answer": "5"}]
    assert clean_tsdkd_source(filtered) == (filtered, {"rows": 1, "kept": 1, "already_filtered": True})
    kept, stats = clean_tsdkd_source([release_row(MATH_COT_PREFIX + "\nQ", RESPONSE), release_row("Other\nQ", RESPONSE)])
    assert len(kept) == 1 and stats["math_cot"] == 1


def test_first_rows_keeps_the_dataset_order_and_zero_means_all():
    from prepare_sra_data import first_rows

    rows = [{"id": i} for i in range(50)]
    assert [r["id"] for r in first_rows(rows, 10)] == list(range(10))
    assert first_rows(rows, 0) == rows and first_rows(rows, 80) == rows
