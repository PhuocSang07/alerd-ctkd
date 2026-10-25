import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataCollator, LLMDataset, text_hash  # noqa: E402
from llm_train import load_tokenizer  # noqa: E402
from span_index import build_rows, sample_spans, span_row, truncation_flags  # noqa: E402


def make_setup(tmp_path_factory, student_kind, teacher_kind, student_pad=True):
    rng = random.Random(0)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student = train_tokenizer(student_kind, corpus, pad=student_pad)
    teacher = train_tokenizer(teacher_kind, corpus, vocab_size=420 if teacher_kind == "bpe" else 120)
    rows = make_rows(rng, 6)
    path = tmp_path_factory.mktemp("data") / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return student, teacher, LLMDataset(str(path), student, teacher, prompt_max_len=64)


# the student/teacher pairs of the experiments: a GPT-2 style byte-level tokenizer without a pad token, and a Llama 3
# style one with a pad token, each against the teacher's tokenizer; plus a word-piece teacher whose newline-less
# tokenization is the hard case for the alignment
@pytest.fixture(scope="module", params=["bpe-bpe", "gpt2-bpe", "bpe-wordpiece"])
def setup(request, tmp_path_factory):
    student_kind, teacher_kind = request.param.split("-")
    if student_kind == "gpt2":
        return make_setup(tmp_path_factory, "bpe", "bpe", student_pad=False)
    return make_setup(tmp_path_factory, student_kind, teacher_kind)


def collator_spans(student, teacher, dataset, indices, max_len=256):
    collator = LLMDataCollator(student, teacher, do_train=True, max_len=max_len, pad_to_multiple_of=4)
    batch = [dataset[i] for i in indices]
    student_inputs, _, _ = collator(batch)
    spans = []
    for b in range(len(indices)):
        safe, mask = student_inputs["pooler_safe_idx"][b], student_inputs["pooler_mask"][b]
        spans.append([[int(safe[s][mask[s]].min()), int(safe[s][mask[s]].max()) + 1] for s in range(len(safe)) if mask[s].any()])
    return student_inputs, spans


def padded_tokenizers(student, teacher):
    """The trainer's tokenizers: pad token added when missing, right padding."""
    for tokenizer in (student, teacher):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return student, teacher


def test_single_sample_spans_equal_padded_batch_spans(setup):
    student, teacher, dataset = setup
    student, teacher = padded_tokenizers(student, teacher)
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
    student, teacher = padded_tokenizers(student, teacher)
    _, alone = collator_spans(student, teacher, dataset, [2])
    _, shuffled = collator_spans(student, teacher, dataset, [5, 2, 0])
    assert alone[0] == shuffled[1]


def test_span_bounds_are_token_boundaries_on_both_sides(setup):
    """Spans end at the same character on both sides, although the teacher's prompt has another length."""
    student, teacher, dataset = setup
    for example in dataset.dataset:
        assert len(example["prompt"]) != len(example["teacher_prompt"])
        bounds, s_off = sample_spans(example, student, teacher, 256)
        teacher_text = example["teacher_prompt"] + example["output"]
        t_off = teacher(teacher_text, add_special_tokens=False, truncation=True, max_length=256,
                        return_offsets_mapping=True)["offset_mapping"]
        shift = len(example["prompt"]) - len(example["teacher_prompt"])
        row = span_row(example, bounds, s_off)
        for char_end, (_, s_end), (_, t_end) in zip(row["char_ends"], row["student_tokens"], row["teacher_tokens"]):
            assert s_off[s_end - 1][1] == char_end == t_off[t_end - 1][1] + shift


def test_spans_cover_the_response_contiguously(setup):
    student, teacher, dataset = setup
    for example in dataset.dataset:
        bounds, s_off = sample_spans(example, student, teacher, 256)
        row = span_row(example, bounds, s_off)
        tokens = row["student_tokens"]
        assert tokens[0][0] == example["s_prompt_len"] and row["teacher_tokens"][0][0] == example["t_prompt_len"]
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
        assert len(steps) == len(row["char_ends"]) and row["n_steps"] == n_steps
        assert all(0 <= k < n_steps for k in steps)
        assert steps == sorted(steps)
        prompt_len = len(example["prompt"])
        for k, char_end, crosses in zip(steps, row["char_ends"], row["crosses_step"]):
            lo = prompt_len + (example["step_char_ends"][k - 1] if k else 0)
            assert char_end - 1 >= lo  # the span's last character sits in step k
            if not crosses:
                assert char_end - 1 < prompt_len + example["step_char_ends"][k] or k == n_steps - 1


def test_rows_carry_the_hash_of_each_models_text(setup):
    student, teacher, dataset = setup
    for example, row in zip(dataset.dataset, build_rows(dataset, student, teacher, 256)):
        assert row["hash"] == text_hash(example["prompt"] + example["output"])
        assert row["teacher_hash"] == text_hash(example["teacher_prompt"] + example["output"])
        assert row["hash"] != row["teacher_hash"]


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


def test_truncation_flags_report_samples_that_lose_the_boxed_answer(setup):
    student, teacher, dataset = setup
    example = dict(dataset.dataset[0])
    example["output"] = example["output"] + " so the answer is \\boxed{7}."
    assert truncation_flags(example, student, teacher, 256) == (False, False)
    assert truncation_flags(example, student, teacher, 30) == (True, True)


def test_trainer_tokenizer_loading_adds_a_pad_token_and_right_padding(setup, tmp_path):
    student, _, _ = setup
    saved = tmp_path / "tok"
    student.save_pretrained(saved)
    loaded = load_tokenizer("gpt2", str(saved), {})
    assert loaded.pad_token is not None and loaded.padding_side == "right" and loaded.eos_token == "<eos>"


def test_collator_without_a_teacher_returns_student_inputs_and_labels_only(setup):
    student, teacher, dataset = setup
    student, _ = padded_tokenizers(student, teacher)
    collator = LLMDataCollator(student, None, do_train=True, max_len=256, pad_to_multiple_of=1)
    inputs, teacher_inputs, labels = collator([dataset[0], dataset[1]])
    assert teacher_inputs is None and "pooler_safe_idx" not in inputs and inputs["example_ids"].tolist() == [0, 1]
    eos = student.eos_token_id
    n = inputs["attention_mask"].sum(1)
    assert all(labels[i, n[i]] == eos for i in range(2))
