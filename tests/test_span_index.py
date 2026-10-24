import json
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataCollator, LLMDataset  # noqa: E402
from prepare_sra_data import step_char_ends  # noqa: E402
from span_index import build_rows, sample_spans, span_row, truncation_flags  # noqa: E402

@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = random.Random(0)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student, teacher = train_tokenizer("bpe", corpus), train_tokenizer("wordpiece", corpus)
    rows = make_rows(rng, 6)
    path = tmp_path_factory.mktemp("data") / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = LLMDataset(str(path), student, teacher, prompt_max_len=64)
    return student, teacher, dataset


def collator_spans(student, teacher, dataset, indices, max_len=256):
    collator = LLMDataCollator(student, teacher, do_train=True, max_len=max_len, pad_to_multiple_of=4)
    batch = [dataset[i] for i in indices]
    student_inputs, _, _ = collator(batch)
    spans = []
    for b in range(len(indices)):
        safe, mask = student_inputs["pooler_safe_idx"][b], student_inputs["pooler_mask"][b]
        spans.append([[int(safe[s][mask[s]].min()), int(safe[s][mask[s]].max()) + 1] for s in range(len(safe)) if mask[s].any()])
    return student_inputs, spans


def test_single_sample_spans_equal_padded_batch_spans(setup):
    student, teacher, dataset = setup
    indices = list(range(len(dataset)))
    inputs, batch_spans = collator_spans(student, teacher, dataset, indices)
    assert inputs["attention_mask"].sum(1).min() < inputs["attention_mask"].shape[1]  # the batch really is padded
    assert inputs["example_ids"].tolist() == indices
    for i in indices:
        bounds, offsets = sample_spans(dataset.dataset[i], student, teacher, 256)
        row = span_row(dataset.dataset[i], bounds, offsets)
        assert row["student_tokens"] == batch_spans[i]


def test_batch_composition_does_not_change_spans(setup):
    student, teacher, dataset = setup
    _, alone = collator_spans(student, teacher, dataset, [2])
    _, shuffled = collator_spans(student, teacher, dataset, [5, 2, 0])
    assert alone[0] == shuffled[1]


def test_span_bounds_are_token_boundaries_on_both_sides(setup):
    student, teacher, dataset = setup
    for example in dataset.dataset:
        bounds, s_off = sample_spans(example, student, teacher, 256)
        full = example["prompt"] + example["output"]
        t_off = teacher(full, truncation=True, max_length=256, return_offsets_mapping=True)["offset_mapping"]
        row = span_row(example, bounds, s_off)
        for char_end, (_, s_end), (_, t_end) in zip(row["char_ends"], row["student_tokens"], row["teacher_tokens"]):
            assert s_off[s_end - 1][1] == char_end == t_off[t_end - 1][1]


def test_spans_cover_the_response_contiguously(setup):
    student, teacher, dataset = setup
    for example in dataset.dataset:
        bounds, s_off = sample_spans(example, student, teacher, 256)
        row = span_row(example, bounds, s_off)
        tokens = row["student_tokens"]
        assert tokens[0][0] == example["s_prompt_len"]
        assert all(a[1] == b[0] for a, b in zip(tokens, tokens[1:]))
        full = example["prompt"] + example["output"]
        tail = full[row["char_ends"][-1]:]  # tokens past the last match have no teacher counterpart
        assert tail.strip() == ""


def test_each_span_belongs_to_exactly_one_step(setup):
    student, teacher, dataset = setup
    rows = build_rows(dataset, student, teacher, 256)
    for example, row in zip(dataset.dataset, rows):
        n_steps = len(example["step_char_ends"])
        steps = row["step_of_span"]
        assert len(steps) == len(row["char_ends"])
        assert all(0 <= k < n_steps for k in steps)
        assert steps == sorted(steps)
        prompt_len = len(example["prompt"])
        for k, char_end, crosses in zip(steps, row["char_ends"], row["crosses_step"]):
            lo = prompt_len + (example["step_char_ends"][k - 1] if k else 0)
            assert char_end - 1 >= lo  # the span's last character sits in step k
            if not crosses:
                assert char_end - 1 < prompt_len + example["step_char_ends"][k] or k == n_steps - 1


def test_max_spans_caps_without_default_subsampling(setup):
    student, teacher, dataset = setup
    full, _ = sample_spans(dataset.dataset[0], student, teacher, 256)
    capped, _ = sample_spans(dataset.dataset[0], student, teacher, 256, max_spans=3)
    assert len(full) - 1 > 3 and len(capped) - 1 == 3


def test_left_padding_is_rejected(setup):
    student, teacher, _ = setup
    student.padding_side = "left"
    try:
        with pytest.raises(ValueError):
            LLMDataCollator(student, teacher)
    finally:
        student.padding_side = "right"


def test_step_char_ends_follow_student_offsets(setup):
    student, _, dataset = setup
    example = dataset.dataset[0]
    ids = student(example["output"], add_special_tokens=False)["input_ids"]
    record = {"id": 0, "response": example["output"], "input_ids": [1, 2] + ids, "response_token_span": [2, 2 + len(ids)],
              "steps": [{"token_start": 2, "token_end": 2 + len(ids)}]}
    assert step_char_ends(student, record) == [len(example["output"])]


def test_truncation_flags_report_samples_that_lose_the_boxed_answer(setup):
    student, teacher, dataset = setup
    example = dict(dataset.dataset[0])
    example["output"] = example["output"] + " so the answer is \\boxed{7}."
    assert truncation_flags(example, student, teacher, 256) == (False, False)
    assert truncation_flags(example, student, teacher, 30) == (True, True)
