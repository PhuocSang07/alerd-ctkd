import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_gain_file, make_rows, make_text, train_tokenizer  # noqa: E402
from data_utils import LLMDataset  # noqa: E402
from span_index import build_rows  # noqa: E402
from token_gain import GainFile, main, pack_gains, span_gains, token_record  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = random.Random(3)
    corpus = [make_text(rng, 6) for _ in range(200)]
    student, teacher = train_tokenizer("bpe", corpus), train_tokenizer("bpe", corpus, vocab_size=420)
    rows = make_rows(rng, 8)
    path = tmp_path_factory.mktemp("gain") / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index = build_rows(LLMDataset(str(path), student, teacher), student, teacher, 256)
    return {"rows": rows, "student": student, "teacher": teacher, "index": index, "rng": rng, "dir": path.parent}


def test_token_record_is_the_joint_tokenization_with_one_probe_per_response_token(world):
    tokenizer = world["student"]
    for row in world["rows"]:
        record = token_record(tokenizer, row, "student")
        full = row["prompt"] + row["output"]
        assert record["input_ids"] == tokenizer(full, add_special_tokens=False)["input_ids"]
        start, end = record["response_token_span"]
        assert start == len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) and end == len(record["input_ids"])
        assert [s["token_end"] for s in record["steps"]] == list(range(start + 1, end + 1))
        lines = record["line_steps"]
        assert len(lines) == len(row["step_char_ends"]) and lines[0]["token_start"] == start
        assert all(a["token_end"] == b["token_start"] for a, b in zip(lines, lines[1:])) and lines[-1]["token_end"] == end
        assert record["response"] == row["output"] and record["id"] == row["id"]


def test_the_teacher_side_scores_the_teachers_prompt(world):
    row = world["rows"][0]
    student, teacher = (token_record(world["student"], row, "student"), token_record(world["teacher"], row, "teacher"))
    assert student["prompt"] == row["prompt"] and teacher["prompt"] == row["teacher_prompt"]
    assert student["hash"] != teacher["hash"]


def test_pack_stores_step_gains_as_sums_of_token_gains(world):
    rng = random.Random(1)
    packed = make_gain_file(world["rows"], world["student"], "student", rng)
    for row in world["rows"]:
        record = token_record(world["student"], row, "student")
        entry = packed["samples"][str(row["id"])]
        assert entry["hash"] == record["hash"] and entry["start"] == record["response_token_span"][0]
        assert len(entry["token_gains"]) == len(record["steps"])
        for step, total in zip(record["line_steps"], entry["step_gains"]):
            a, b = step["token_start"] - entry["start"], step["token_end"] - entry["start"]
            assert total == pytest.approx(sum(entry["token_gains"][a:b]))
        assert sum(entry["step_gains"]) == pytest.approx(sum(entry["token_gains"]))


def test_pack_rejects_missing_or_misaligned_gains(world):
    records = [token_record(world["student"], world["rows"][0], "student")]
    with pytest.raises(ValueError, match="no gains"):
        pack_gains(records, {}, "m", "m")
    with pytest.raises(ValueError, match="gains for"):
        pack_gains(records, {str(records[0]["id"]): [0.0, 1.0]}, "m", "m")


@pytest.mark.parametrize("side,tokenizer_key", [("student", "student"), ("teacher", "teacher")])
def test_span_gains_add_up_to_the_step_gain(world, side, tokenizer_key):
    """With no span across a step boundary, the spans of a step sum to the gain of that step, for either model."""
    packed = make_gain_file(world["rows"], world[tokenizer_key], side, random.Random(2))
    checked = 0
    for row in world["index"]:
        if any(row["crosses_step"]):
            continue
        entry = packed["samples"][str(row["id"])]
        example = world["rows"][row["id"]]
        assert row["char_ends"][-1] == len(example["prompt"]) + len(example["output"])  # spans reach the last token
        per_span = span_gains(entry, row[f"{side}_tokens"])
        per_step = [0.0] * row["n_steps"]
        for gain, step in zip(per_span, row["step_of_span"]):
            per_step[step] += gain
        assert per_step == pytest.approx(entry["step_gains"], abs=1e-9)
        assert sum(per_span) == pytest.approx(sum(entry["token_gains"]))
        checked += 1
    assert checked >= 5


def test_a_span_gain_is_the_difference_of_the_answer_log_probability_at_its_ends():
    """g_t = b_t - b_{t-1}, so the span [a, b) has gain b_b - b_a: a telescoping sum."""
    b = [-6.0, -5.5, -5.9, -3.0, -3.1, -0.4]  # log-probability of the answer after t response tokens (t = 0..5)
    entry = {"start": 7, "token_gains": [b[t] - b[t - 1] for t in range(1, 6)]}
    assert span_gains(entry, [(7, 9), (9, 10), (10, 12)]) == pytest.approx([b[2] - b[0], b[3] - b[2], b[5] - b[3]])


def test_gain_file_checks_hash_start_length_and_ids(world, tmp_path):
    packed = make_gain_file(world["rows"], world["student"], "student", random.Random(4), model="tiny-student")
    path = tmp_path / "gain-tokens-student.json"
    path.write_text(json.dumps(packed))
    gains = GainFile(str(path), "student")
    assert gains.model == "tiny-student" and gains.tokenizer == "tiny-student"
    row = world["index"][0]
    first, last = row["student_tokens"][0][0], row["student_tokens"][-1][1]
    assert gains.entry(row["id"], row["hash"], first, last)["start"] == first
    with pytest.raises(ValueError, match="hash mismatch"):
        gains.entry(row["id"], "0" * 40, first, last)
    with pytest.raises(ValueError, match="response start or length"):
        gains.entry(row["id"], row["hash"], first + 1, last)
    with pytest.raises(ValueError, match="response start or length"):
        gains.entry(row["id"], row["hash"], first, last + 10_000)
    with pytest.raises(ValueError, match="no student gains for id 999"):
        gains.entry(999, row["hash"], first, last)


def test_a_missing_or_foreign_file_stops_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="score_gain.sh"):
        GainFile(str(tmp_path / "nope.json"), "teacher")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"0": [0.1]}))  # the raw answer_gain output, not a packed file
    with pytest.raises(ValueError, match="not a gain-tokens file"):
        GainFile(str(bad), "teacher")


def test_cli_records_then_pack(world, tmp_path):
    data = tmp_path / "train-sra.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in world["rows"][:3]), encoding="utf-8")
    world["student"].save_pretrained(tmp_path / "tok")
    main(["records", "--data", str(data), "--tokenizer", str(tmp_path / "tok"), "--side", "student",
          "--output", str(tmp_path / "tokens.jsonl")])
    records = [json.loads(line) for line in open(tmp_path / "tokens.jsonl", encoding="utf-8")]
    raw = {str(r["id"]): [0.5] * len(r["steps"]) for r in records}
    (tmp_path / "raw.json").write_text(json.dumps(raw))
    main(["pack", "--records", str(tmp_path / "tokens.jsonl"), "--gains", str(tmp_path / "raw.json"),
          "--model", "m", "--tokenizer", "t", "--output", str(tmp_path / "gain-tokens-m.json")])
    packed = json.loads((tmp_path / "gain-tokens-m.json").read_text())
    assert set(packed["samples"]) == {"0", "1", "2"} and packed["tokenizer"] == "t"


def test_records_feed_sgl_answer_gain_and_its_answer_is_the_gsm8k_answer(world):
    src = os.environ.get("SGL_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set SGL_SRC to compare with sgl")
    sys.path.insert(0, src)
    try:
        from sgl.eval.graders.answer_scoring import extract_boxed
        from sgl.signals.answer_gain import PROBE
    except ImportError as error:
        pytest.skip(f"sgl is not importable here: {error}")
    finally:
        sys.path.remove(src)
    assert PROBE == "\n\nTherefore, the final answer is \\boxed{"
    for row in world["rows"]:
        record = token_record(world["student"], row, "student")
        assert extract_boxed(record["response"]) == row["answer"]
        # the keys answer_gain reads: id, response, input_ids, response_token_span, steps[*].token_end
        cuts = [record["response_token_span"][0]] + [s["token_end"] for s in record["steps"]]
        assert cuts == list(range(record["response_token_span"][0], len(record["input_ids"]) + 1))
