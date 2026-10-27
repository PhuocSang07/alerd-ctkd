import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def run(script, env, check=True, *args):
    return subprocess.run([BASH, f"scripts/{script}", *args], cwd=ROOT, env=dict(os.environ, PYTHON="echo", **env),
                          capture_output=True, text=True, check=check)


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("train-sra.jsonl", "spans-Qwen2.5-14B-Instruct.jsonl", "span-gains.json"):
        (folder / name).write_text("{}")
    return folder


def commands(result, marker):
    return [line for line in result.stdout.splitlines() if marker in line]


def flag(command, name):
    words = command.split()
    return words[words.index(name) + 1]


def test_each_experiment_gets_its_weights_and_the_teacher_only_when_it_distills(tmp_path, data_dir):
    env = {"DATA_DIR": str(data_dir), "OUT_ROOT": str(tmp_path / "out"), "EXPERIMENTS": "sft kd-token gain-kd gain-both-lam1"}
    lines = commands(run("span_kd_train.sh", env), "span_kd_train.py")
    assert len(lines) == 4
    sft, kd_token, gain_kd, lam1 = lines
    assert "--teacher-model" not in sft and flag(sft, "--kd-beta") == "0"
    assert flag(kd_token, "--ce-weights") == flag(kd_token, "--kd-weights") == "token"
    assert flag(kd_token, "--teacher-model") == "Qwen/Qwen2.5-14B-Instruct"
    assert (flag(gain_kd, "--ce-weights"), flag(gain_kd, "--kd-weights")) == ("token", "diff")
    assert (flag(lam1, "--ce-weights"), flag(lam1, "--kd-weights"), flag(lam1, "--alloc-lambda")) == ("student", "diff", "1.0")
    assert all("--dump-spans" in line and flag(line, "--seed") == "10" for line in lines)


def test_default_experiments_are_the_five_of_the_ablation(tmp_path, data_dir):
    lines = commands(run("span_kd_train.sh", {"DATA_DIR": str(data_dir), "OUT_ROOT": str(tmp_path / "out")}),
                     "span_kd_train.py")
    assert [flag(line, "--output-dir").rsplit("/", 1)[-1] for line in lines] == [
        "sft", "kd-token", "gain-ce", "gain-kd", "gain-both"]
    assert flag(lines[-1], "--batch-size") == "32" and flag(lines[-1], "--micro-batch-size") == "4"


def test_training_stops_without_the_scoring_outputs_or_with_an_unknown_experiment(tmp_path, data_dir):
    missing = run("span_kd_train.sh", {"DATA_DIR": str(tmp_path / "nothing")}, check=False)
    assert missing.returncode == 2 and "score_gain.sh" in missing.stderr
    unknown = run("span_kd_train.sh", {"DATA_DIR": str(data_dir), "OUT_ROOT": str(tmp_path / "o"),
                                       "EXPERIMENTS": "gain-everything"}, check=False)
    assert unknown.returncode == 2 and "unknown experiment" in unknown.stderr


def test_evaluation_runs_both_suites_at_4096_tokens_for_every_checkpoint(tmp_path):
    for name in ("sft", "gain-both"):
        (tmp_path / "out" / name).mkdir(parents=True)
    env = {"OUT_ROOT": str(tmp_path / "out"), "RESULTS_DIR": str(tmp_path / "res"), "EXPERIMENTS": "sft gain-both",
           "ZERO_SHOT": "1"}
    lines = commands(run("span_kd_eval.sh", env), "benchmark_eval.py")
    assert len(lines) == 6  # (zero-shot, sft, gain-both) x (gsm8k+math500, aime24+aime25)
    assert [flag(line, "--benchmarks") for line in lines[:2]] == ["gsm8k,math500", "aime24,aime25"]
    assert [flag(line, "--n-samples") for line in lines[:2]] == ["3", "16"]
    assert all(flag(line, "--max-tokens") == "4096" and flag(line, "--max-model-len") == "6144" for line in lines)
    assert flag(lines[0], "--model") == "Qwen/Qwen2.5-1.5B-Instruct" and flag(lines[2], "--model").endswith("/sft")
    assert flag(lines[5], "--results-dir").endswith("res/gain-both")


def test_evaluation_needs_the_checkpoints(tmp_path):
    result = run("span_kd_eval.sh", {"OUT_ROOT": str(tmp_path), "EXPERIMENTS": "sft"}, check=False)
    assert result.returncode == 2 and "span_kd_train.sh" in result.stderr


def test_run_all_on_the_tsdkd_math_data_trains_two_epochs_with_batch_64(tmp_path):
    data = tmp_path / "w-tsdkd-limit50" / "Qwen2.5-1.5B-Instruct"
    data.mkdir(parents=True)
    for name in ("train-sra.jsonl", "spans-Qwen2.5-14B-Instruct.jsonl", "span-gains.json"):
        (data / name).write_text("{}")
    env = {"DATASET": "tsdkd", "LIMIT": "50", "SKIP_SCORING": "1", "EXPERIMENTS": "gain-both",
           "WORK_ROOT": str(tmp_path / "w"), "OUT_ROOT": str(tmp_path / "out"), "RESULTS_DIR": str(tmp_path / "res")}
    result = run("span_kd_run_all.sh", env, check=False)
    [line] = commands(result, "span_kd_train.py")
    assert flag(line, "--epochs") == "2" and flag(line, "--batch-size") == "64" and flag(line, "--micro-batch-size") == "4"


def test_run_all_scores_with_word_spans_then_trains_then_evaluates(tmp_path):
    (tmp_path / "w-s1k-limit50").mkdir()
    (tmp_path / "w-s1k-limit50" / "s1k-clean.jsonl").write_text("{}\n" * 60)
    env = {"DATASET": "s1k", "LIMIT": "50", "EXPERIMENTS": "sft", "WORK_ROOT": str(tmp_path / "w"),
           "OUT_ROOT": str(tmp_path / "out"), "RESULTS_DIR": str(tmp_path / "res")}
    # echo stands in for Python, so the training script finds no scoring outputs: it must say so after scoring ran
    result = run("span_kd_run_all.sh", env, check=False)
    assert any("clean-s1k" in line and "w-s1k-limit50" in line for line in result.stdout.splitlines())
    assert result.returncode == 2 and "w-s1k-limit50/Qwen2.5-1.5B-Instruct/train-sra.jsonl" in result.stderr
