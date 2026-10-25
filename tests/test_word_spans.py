import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataCollator, LLMDataset, text_hash  # noqa: E402
from span_index import build_rows, main as span_index_main  # noqa: E402
from token_gain import GainFile, main as token_gain_main, span_gains, span_signals  # noqa: E402
from word_spans import word_end_offsets, word_pair_bounds, word_span_starts  # noqa: E402

SENTENCE = "Natalia sold 48/2 = 24 clips in May."
SENTENCE_SPANS = ["Natalia", " sold", " 48", "/", "2", " =", " 24", " clips", " in", " May", "."]


def words_of(text):
    ends = [0] + [int(e) for e in word_end_offsets(text)[:, 1]]
    return [text[a:b] for a, b in zip(ends, ends[1:])]


def spans_of(tokenizer, prompt, output):
    encoded = tokenizer(prompt + output, add_special_tokens=False, return_offsets_mapping=True)
    start = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    starts = word_span_starts(encoded["offset_mapping"], prompt, output, start)
    return encoded, start, starts


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer is not cached")


@pytest.fixture(scope="module")
def tokenizer():
    rng = random.Random(0)
    return train_tokenizer("bpe", [make_text(rng, 6) for _ in range(200)])


# ---- the words ---------------------------------------------------------------------------------------------------

def test_words_are_numbers_word_runs_and_single_symbols_with_their_leading_space():
    assert words_of(SENTENCE) == SENTENCE_SPANS
    assert words_of("paid $1,200 and 3.5 hours") == ["paid", " $", "1,200", " and", " 3.5", " hours"]
    assert words_of("It is 24.") == ["It", " is", " 24", "."]  # a final period is not part of the number
    assert words_of("\\boxed{72}") == ["\\", "boxed", "{", "72", "}"]


def test_a_line_break_belongs_to_the_last_word_of_its_line():
    assert words_of("a b.\nc d\n\ne") == ["a", " b", ".\n", "c", " d\n\n", "e"]
    assert word_end_offsets("").shape == (0, 2)


# ---- the spans ---------------------------------------------------------------------------------------------------

def test_the_sample_sentence_is_eleven_spans_with_the_qwen_tokenizer(qwen):
    prompt = "<|im_start|>assistant\n"
    encoded, start, starts = spans_of(qwen, prompt, SENTENCE)
    text = prompt + SENTENCE
    offsets = encoded["offset_mapping"]
    pieces = [text[offsets[a][0]:offsets[b - 1][1]] for a, b in zip(starts, starts[1:])]
    assert pieces == SENTENCE_SPANS


@pytest.mark.parametrize("text", ["It cost $1,200 in total.", "He ran 3.5 miles.", "So \\boxed{1,200}."])
def test_money_and_decimals_stay_single_spans(qwen, text):
    prompt = "<|im_start|>assistant\n"
    encoded, _, starts = spans_of(qwen, prompt, text)
    offsets, full = encoded["offset_mapping"], prompt + text
    pieces = [full[offsets[a][0]:offsets[b - 1][1]].strip() for a, b in zip(starts, starts[1:])]
    for number in ("1,200", "3.5"):
        if number in text:
            assert number in pieces


def test_every_response_token_is_in_exactly_one_span(tokenizer):
    for row in make_rows(random.Random(1), 12):
        encoded, start, starts = spans_of(tokenizer, row["prompt"], row["output"])
        assert starts[0] == start and starts[-1] == len(encoded["input_ids"])
        assert all(a < b for a, b in zip(starts, starts[1:]))  # no empty span, none out of order
        covered = [t for a, b in zip(starts, starts[1:]) for t in range(a, b)]
        assert covered == list(range(start, len(encoded["input_ids"])))


def test_trailing_whitespace_after_the_last_word_is_covered(tokenizer):
    encoded, start, starts = spans_of(tokenizer, "Q: x\nA:\n", "so 2 .  \n")
    assert starts[-1] == len(encoded["input_ids"])


def test_spans_never_cross_a_step_boundary(tokenizer, tmp_path):
    rng = random.Random(2)
    rows = make_rows(rng, 10)
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(path), tokenizer, tokenizer, prompt_max_len=64)
    index = build_rows(dataset, tokenizer, tokenizer, 256, span_mode="word")
    assert all(not any(r["crosses_step"]) for r in index)
    for row, source in zip(index, rows):
        assert row["n_steps"] == len(source["step_char_ends"]) and set(row["step_of_span"]) == set(range(row["n_steps"]))


def test_word_spans_need_one_tokenizer_for_both_models(tokenizer):
    encoded = tokenizer("Q: x\nA:\nso 2.", add_special_tokens=False, return_offsets_mapping=True)
    with pytest.raises(ValueError, match="one tokenizer"):
        word_pair_bounds(encoded["offset_mapping"], len(encoded["input_ids"]) + 3, 4, 4, "Q: x\nA:\n", "so 2.")


# ---- the same spans offline and in the collator -------------------------------------------------------------------

def test_offline_spans_equal_the_spans_of_the_collator(tokenizer, tmp_path):
    rows = make_rows(random.Random(3), 8)  # the teacher's prompt is longer than the student's
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(path), tokenizer, tokenizer, prompt_max_len=64)
    tokenizer.padding_side = "right"
    index = build_rows(dataset, tokenizer, tokenizer, 256, span_mode="word")

    collator = LLMDataCollator(tokenizer, tokenizer, do_train=True, max_len=256, pad_to_multiple_of=4, span_mode="word")
    student, teacher, _ = collator([dataset[i] for i in range(len(rows))])
    assert student["attention_mask"].sum(1).min() < student["attention_mask"].shape[1]  # the batch really is padded
    for b, row in enumerate(index):
        ranges = {}
        for side, inputs in (("student", student), ("teacher", teacher)):
            safe, mask = inputs["pooler_safe_idx"][b], inputs["pooler_mask"][b]
            ranges[side] = [[int(safe[s][mask[s]].min()), int(safe[s][mask[s]].max()) + 1]
                            for s in range(len(safe)) if mask[s].any()]
        assert ranges["student"] == row["student_tokens"] and ranges["teacher"] == row["teacher_tokens"]
        assert len(ranges["student"]) == len(row["char_ends"])
        example = dataset.dataset[b]
        assert row["hash"] == text_hash(example["prompt"] + example["output"])
        assert row["teacher_hash"] == text_hash(example["teacher_prompt"] + example["output"])
        # the teacher sees the same response tokens under its own, longer prompt
        shift = example["t_prompt_len"] - example["s_prompt_len"]
        assert shift > 0 and row["teacher_tokens"] == [[a + shift, e + shift] for a, e in row["student_tokens"]]


def test_the_collator_rejects_an_unknown_span_mode(tokenizer):
    with pytest.raises(ValueError, match="span_mode"):
        LLMDataCollator(tokenizer, tokenizer, span_mode="phrase")


# ---- the gains of the spans ---------------------------------------------------------------------------------------

def test_span_gains_add_up_to_the_gain_of_the_whole_response(tokenizer, tmp_path):
    rng = random.Random(4)
    rows = make_rows(rng, 6)
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(path), tokenizer, tokenizer, prompt_max_len=64)
    index = build_rows(dataset, tokenizer, tokenizer, 256, span_mode="word")
    files = {side: make_gain_file(rows, tokenizer, side, rng) for side in ("student", "teacher")}
    for row, source in zip(index, rows):
        entry = files["student"]["samples"][str(row["id"])]
        gains = span_gains(entry, row["student_tokens"])
        assert len(gains) == len(row["char_ends"])
        b = [0.0]  # b_t - b_start by token: the cumulative token gains
        for g in entry["token_gains"]:
            b.append(b[-1] + g)
        assert sum(gains) == pytest.approx(b[-1] - b[0])
        ends = [e - entry["start"] for _, e in row["student_tokens"]]
        assert gains == pytest.approx([b[e] - b[s] for s, e in zip([0] + ends[:-1], ends)])


def test_signals_are_the_student_gain_and_teacher_minus_student(tokenizer, tmp_path):
    rng = random.Random(5)
    rows = make_rows(rng, 5)
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(data), tokenizer, tokenizer, prompt_max_len=64)
    index = build_rows(dataset, tokenizer, tokenizer, 256, span_mode="word")
    for side in ("student", "teacher"):
        (tmp_path / f"{side}.json").write_text(json.dumps(make_gain_file(rows, tokenizer, side, rng)))
    (tmp_path / "spans.jsonl").write_text("\n".join(json.dumps(r) for r in index))
    token_gain_main(["signals", "--spans", str(tmp_path / "spans.jsonl"), "--student-gains",
                     str(tmp_path / "student.json"), "--teacher-gains", str(tmp_path / "teacher.json"),
                     "--output", str(tmp_path / "signals.json")])
    signals = json.loads((tmp_path / "signals.json").read_text())
    assert set(signals) == {str(r["id"]) for r in rows}
    student, teacher = GainFile(tmp_path / "student.json", "student"), GainFile(tmp_path / "teacher.json", "teacher")
    for row in index:
        one = signals[str(row["id"])]
        assert len(one["student"]) == len(one["teacher"]) == len(one["diff"]) == len(row["char_ends"])
        assert one["diff"] == pytest.approx([t - s for t, s in zip(one["teacher"], one["student"])])
        assert one == span_signals(row, student, teacher)
        # same tokens on both sides, so the gains differ only by the models' scores of the same text
        assert row["teacher_tokens"][0][1] - row["teacher_tokens"][0][0] == row["student_tokens"][0][1] - row["student_tokens"][0][0]


def test_signals_stop_when_the_gains_were_scored_on_another_text(tokenizer, tmp_path):
    rng = random.Random(6)
    rows = make_rows(rng, 3)
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(data), tokenizer, tokenizer, prompt_max_len=64)
    index = build_rows(dataset, tokenizer, tokenizer, 256, span_mode="word")
    changed = [dict(r, output=r["output"].replace("the", "a")) for r in rows]
    (tmp_path / "student.json").write_text(json.dumps(make_gain_file(changed, tokenizer, "student", rng)))
    (tmp_path / "teacher.json").write_text(json.dumps(make_gain_file(rows, tokenizer, "teacher", rng)))
    with pytest.raises(ValueError, match="different text|different response"):
        span_signals(index[0], GainFile(tmp_path / "student.json", "student"),
                     GainFile(tmp_path / "teacher.json", "teacher"))


def test_span_index_cli_word_mode_reports_spans_tokens_and_probes(tokenizer, tmp_path, capsys):
    rows = make_rows(random.Random(7), 4)
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tokenizer.save_pretrained(tmp_path / "tok")
    span_index_main(["--data", str(data), "--student-tokenizer", str(tmp_path / "tok"),
                     "--teacher-tokenizer", str(tmp_path / "tok"), "--span-mode", "word", "--max-len", "256",
                     "--output", str(tmp_path / "spans.jsonl")])
    out = capsys.readouterr().out
    assert "tokens/span" in out and "answer-gain probes" in out and "spans crossing a step boundary: 0/" in out
