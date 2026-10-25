import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import CHAT_TEMPLATE, make_text, train_tokenizer  # noqa: E402
from prepare_sra_data import build_rows, check_boundary, clean_rows, convert_row, main  # noqa: E402
from prompts import FIXED_DATE, INSTRUCTION, build_prompt, has_chat_template  # noqa: E402
from benchmark_eval import build_prompts, gsm8k_gold, gsm8k_records  # noqa: E402

GSM8K_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
              "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
}


def test_convert_row_drops_annotations_and_boxes_the_answer():
    row = convert_row(GSM8K_ROW)
    assert row["output"] == ("Natalia sold 48/2 = 24 clips in May.\n"
                             "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
                             "The final answer is \\boxed{72}.")
    assert row["answer"] == "72" and "<<" not in row["output"] and "####" not in row["output"]
    # every line is a step; a step ends after its newline, the last one at the end of the text
    assert row["step_char_ends"] == [37, 96, len(row["output"])]
    lines = row["output"].split("\n")
    assert [row["output"][a:b] for a, b in zip([0] + row["step_char_ends"][:-1], row["step_char_ends"])] == \
        [line + "\n" for line in lines[:-1]] + [lines[-1]]


@pytest.mark.parametrize("raw,expected", [("1,000", "1000"), ("-5", "-5"), ("0.5", "0.5"), ("12,345,678", "12345678")])
def test_final_answer_drops_thousands_separators(raw, expected):
    row = convert_row({"question": "q", "answer": f"step one\n#### {raw}"})
    assert row["answer"] == expected and row["output"].endswith(f"\\boxed{{{expected}}}.")


def test_malformed_rows_are_rejected():
    for answer in ("no marker here", "only the marker\n#### ", "####"):
        with pytest.raises(ValueError):
            convert_row({"question": "q", "answer": answer})


def test_clean_rows_numbers_ids_and_counts_non_numeric_answers():
    rows, stats = clean_rows([GSM8K_ROW, {"question": "q", "answer": "x\n#### 3 apples"}])
    assert [r["id"] for r in rows] == [0, 1] and stats["rows"] == 2 and stats["non_numeric_answers"] == 1


@pytest.fixture(scope="module")
def tokenizers():
    rng = random.Random(0)
    corpus = [make_text(rng, 6) for _ in range(200)]
    gpt2_like = train_tokenizer("bpe", corpus, pad=False)  # no pad token, no chat template
    chat = train_tokenizer("bpe", corpus, vocab_size=420)
    chat.chat_template = CHAT_TEMPLATE
    return gpt2_like, chat


def test_prompt_is_raw_without_a_chat_template_and_a_user_turn_with_one(tokenizers):
    raw, chat = tokenizers
    assert not has_chat_template(raw) and has_chat_template(chat)
    assert build_prompt(raw, "What is 2+2?") == INSTRUCTION.format(question="What is 2+2?")
    assert build_prompt(raw, "q").startswith("Please reason step by step, and put your final answer within \\boxed{}.")
    assert build_prompt(chat, "q") == "<|user|>\n" + INSTRUCTION.format(question="q") + "\n<|assistant|>\n"


def test_date_in_a_chat_template_is_fixed(tokenizers):
    _, chat = tokenizers
    chat.chat_template = "{{ date_string|default('today') }}|{% for m in messages %}{{ m['content'] }}{% endfor %}"
    try:
        assert build_prompt(chat, "q").startswith(FIXED_DATE + "|")
    finally:
        chat.chat_template = CHAT_TEMPLATE


def test_build_rows_wrap_the_same_output_in_two_prompts(tokenizers):
    raw, chat = tokenizers
    clean, _ = clean_rows([GSM8K_ROW])
    (row,) = build_rows(clean, raw, chat)
    assert row["prompt"] != row["teacher_prompt"] and row["output"] == clean[0]["output"]
    assert row["prompt"].endswith(row["question"]) and row["teacher_prompt"].endswith("<|assistant|>\n")


def test_a_token_across_the_prompt_boundary_is_rejected(tokenizers):
    raw, _ = tokenizers
    words = ("let", "area", "sum", "value", "side", "length", "root", "ways", "number")
    straddling = []
    for first in words:
        for second in words:
            check_boundary(raw, f"Q: {first}\n", second, "student")  # a newline always separates them
            try:
                check_boundary(raw, f"Q: {first}", second, "student")  # same pre-token: BPE may merge across
            except ValueError as error:
                assert "straddles" in str(error)
                straddling.append((first, second))
    assert straddling


def test_cli_clean_then_build(tmp_path, tokenizers):
    raw, chat = tokenizers
    source = tmp_path / "gsm8k.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in (GSM8K_ROW, GSM8K_ROW)), encoding="utf-8")
    main(["clean", "--dataset", str(source), "--output", str(tmp_path / "clean.jsonl")])
    raw.save_pretrained(tmp_path / "student")
    chat.save_pretrained(tmp_path / "teacher")
    main(["build", "--clean", str(tmp_path / "clean.jsonl"), "--student-tokenizer", str(tmp_path / "student"),
          "--teacher-tokenizer", str(tmp_path / "teacher"), "--output", str(tmp_path / "work" / "train-sra.jsonl")])
    rows = [json.loads(line) for line in open(tmp_path / "work" / "train-sra.jsonl", encoding="utf-8")]
    assert len(rows) == 2 and {"prompt", "teacher_prompt", "output", "answer", "step_char_ends", "id"} <= set(rows[0])


# ---- evaluation side: the benchmark and the prompt the model is evaluated on ----------------------------------

def test_gsm8k_benchmark_records_have_the_benchmark_format():
    records = gsm8k_records([GSM8K_ROW, {"question": "q2", "answer": "a\n#### 1,250"}])
    assert [r["gold"] for r in records] == ["72", "1250"] and gsm8k_gold("x\n#### 7") == "7"
    assert records[0]["prompt"] == INSTRUCTION.format(question=GSM8K_ROW["question"])
    assert all(r["task_type"] == "math" and set(r) >= {"id", "question", "prompt", "gold"} for r in records)


def test_eval_prompts_equal_the_training_prompts(tokenizers, tmp_path):
    raw, chat = tokenizers
    records = gsm8k_records([GSM8K_ROW])
    for tokenizer in (raw, chat):
        tokenizer.save_pretrained(tmp_path / "m")
        ids = build_prompts("unused", records, {"base_model": str(tmp_path / "m"), "max_model_len": 4096})[0]
        expected = tokenizer.encode(build_prompt(tokenizer, GSM8K_ROW["question"]), add_special_tokens=False)
        assert ids["prompt_token_ids"] == expected


def test_a_prompt_longer_than_the_context_stops_the_run(tokenizers, tmp_path):
    raw, _ = tokenizers
    raw.save_pretrained(tmp_path / "m")
    with pytest.raises(ValueError, match="does not fit"):
        build_prompts("unused", gsm8k_records([GSM8K_ROW]), {"base_model": str(tmp_path / "m"), "max_model_len": 8})
