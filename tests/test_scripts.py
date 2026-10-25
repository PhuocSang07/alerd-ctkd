import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import Arguments  # noqa: E402
from run_distill_llm import parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

METHODS = ["sft", "sra", "sra-uniform", "alerd-ctkd", "alerd-ctkd-gain-student", "alerd-ctkd-gain-teacher",
           "alerd-ctkd-gain-diff", "alerd-ctkd-unit-step", "alerd-ctkd-random", "alerd-ctkd-pred-easy",
           "alerd-ctkd-pred-hard", "alerd-ctkd-answer-only-unit", "alerd-ctkd-corpus", "alerd-ctkd-nonorm",
           "alerd-ctkd-t2-l0", "alerd-ctkd-t2-l05", "alerd-ctkd-t2-l1", "alerd-ctkd-t4-l05"]
STUDENT_DIRS = {"gpt2-xl": "gpt2-xl", "llama-3.2-1b": "Llama-3.2-1B-Instruct"}


def last_value(argv, flag):
    """argparse keeps the last occurrence of a repeated flag."""
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


def make_work(tmp_path, student, gain_files=("student", "teacher")):
    work = tmp_path / "work"
    model_dir = work / STUDENT_DIRS[student]
    teacher_dir = work / "Qwen2.5-7B-Instruct"
    model_dir.mkdir(parents=True)
    teacher_dir.mkdir(parents=True)
    (model_dir / "train-sra.jsonl").write_text("{}\n")
    (model_dir / "predictability.json").write_text("{}")
    if "student" in gain_files:
        (model_dir / f"gain-tokens-{STUDENT_DIRS[student]}.json").write_text("{}")
    if "teacher" in gain_files:
        (teacher_dir / "gain-tokens-Qwen2.5-7B-Instruct.json").write_text("{}")
    return work


def run_train_script(tmp_path, student, method, seed=None, extra_env=None, gain_files=("student", "teacher"),
                     check=True):
    """Run alerd_ctkd_train.sh with `echo` standing in for python; returns (process, arguments it would pass)."""
    work, out = make_work(tmp_path, student, gain_files), tmp_path / "out"
    (work / STUDENT_DIRS[student] / "spans-Qwen2.5-7B-Instruct.jsonl").write_text("{}\n")
    env = dict(os.environ, PYTHON="echo", WORK_DIR=str(work), OUT_ROOT=str(out), **(extra_env or {}))
    command = [BASH, "scripts/alerd_ctkd_train.sh", student, method] + ([str(seed)] if seed else [])
    result = subprocess.run(command, cwd=ROOT, env=env, check=check, capture_output=True, text=True)
    log = out / student / method / f"seed-{seed or 42}" / "train.log"
    return result, (log.read_text().replace("run_distill_llm.py", "").split() if log.exists() else [])


def parse_argv(tmp_path, argv):
    """Parse like run_distill_llm; the data files are not there (echo stands in for the scripts)."""
    argv = list(argv)
    train_data = argv[argv.index("--train_data") + 1]
    argv[argv.index("--train_data") + 1] = str(tmp_path / "train.jsonl")
    (tmp_path / "train.jsonl").write_text("{}\n")
    return parse(argv), train_data


@pytest.mark.parametrize("student,layer", [("gpt2-xl", 48), ("llama-3.2-1b", 16)])
@pytest.mark.parametrize("method", METHODS)
def test_every_method_yields_arguments_the_trainer_accepts(tmp_path, student, layer, method):
    _, argv = run_train_script(tmp_path, student, method)
    (args, extras), train_data = parse_argv(tmp_path, argv)
    assert isinstance(args, Arguments) and train_data.endswith("train-sra.jsonl")
    assert args.num_train_epochs == 3 and args.grad_accum_steps * args.batch_size == 32
    assert args.learning_rate == 5e-5 and args.min_learning_rate == 1e-5 and args.warmup_ratio == 0.1
    assert args.max_len == 512 and args.use_lora is False and args.student_dtype == "float32"
    assert extras.student_model_type == ("gpt2" if student == "gpt2-xl" else "llama")
    if method == "sft":
        assert args.sft_only and args.teacher_layers_mapping == [] and args.teacher_model == ""
    else:
        assert not args.sft_only and args.hard_label_loss_weight == 0.5
        assert args.student_encoder_layers_finetuned == [layer] and args.teacher_layers_mapping == [28]
        assert args.teacher_model == "Qwen/Qwen2.5-7B-Instruct" and args.teacher_embedding_dimension == 3584


@pytest.mark.parametrize("method,source,tau,lam", [
    ("sra", "attention", 1.0, 0.5), ("sra-uniform", "uniform", 1.0, 0.5), ("alerd-ctkd", "alerd", 1.0, 0.5),
    ("alerd-ctkd-random", "random", 1.0, 0.5), ("alerd-ctkd-pred-easy", "predictability-easy", 1.0, 0.5),
    ("alerd-ctkd-pred-hard", "predictability-hard", 1.0, 0.5), ("alerd-ctkd-t2-l0", "alerd", 2.0, 0.0),
    ("alerd-ctkd-t2-l05", "alerd", 2.0, 0.5), ("alerd-ctkd-t2-l1", "alerd", 2.0, 1.0),
    ("alerd-ctkd-t4-l05", "alerd", 4.0, 0.5)])
def test_method_selects_the_expected_weights(tmp_path, method, source, tau, lam):
    _, argv = run_train_script(tmp_path, "llama-3.2-1b", method)
    assert argv[argv.index("--span_weight_source") + 1] == source
    if source not in ("attention", "uniform"):
        assert float(last_value(argv, "--alloc_tau")) == tau
        assert float(last_value(argv, "--alloc_lambda")) == lam


def gain_choice(argv):
    return last_value(argv, "--gain_source"), last_value(argv, "--gain_unit")


def test_gain_source_and_unit_come_from_the_environment_or_the_method(tmp_path):
    assert gain_choice(run_train_script(tmp_path / "a", "gpt2-xl", "alerd-ctkd")[1]) == ("student", "span")
    env = {"GAIN_SOURCE": "diff", "GAIN_UNIT": "step"}
    expected = {"alerd-ctkd": ("diff", "step"), "alerd-ctkd-random": ("diff", "step"),
                "alerd-ctkd-gain-student": ("student", "span"), "alerd-ctkd-gain-teacher": ("teacher", "span"),
                "alerd-ctkd-gain-diff": ("diff", "span"), "alerd-ctkd-unit-step": ("diff", "step")}
    for method, choice in expected.items():
        assert gain_choice(run_train_script(tmp_path / method, "gpt2-xl", method, extra_env=env)[1]) == choice
    unit = run_train_script(tmp_path / "u", "gpt2-xl", "alerd-ctkd-unit-step", extra_env={"GAIN_SOURCE": "teacher"})[1]
    assert gain_choice(unit) == ("teacher", "step")


def test_gain_files_are_passed_and_the_ablation_flags_override_the_defaults(tmp_path):
    _, argv = run_train_script(tmp_path / "a", "llama-3.2-1b", "alerd-ctkd-corpus")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--alloc_norm"] == ["per-trace", "corpus"]
    assert argv[argv.index("--student_gain_file") + 1].endswith("Llama-3.2-1B-Instruct/gain-tokens-Llama-3.2-1B-Instruct.json")
    assert argv[argv.index("--teacher_gain_file") + 1].endswith("Qwen2.5-7B-Instruct/gain-tokens-Qwen2.5-7B-Instruct.json")
    assert "--answer_only_unit" in run_train_script(tmp_path / "b", "gpt2-xl", "alerd-ctkd-answer-only-unit")[1]


def test_sft_passes_no_teacher_and_a_hard_label_weight_of_one(tmp_path):
    _, argv = run_train_script(tmp_path, "gpt2-xl", "sft")
    assert argv[argv.index("--hard_label_loss_weight") + 1] == "1.0"
    assert not any(a.startswith("--teacher") or a in ("--span_weight_source", "--spans_file") for a in argv)


def test_output_directory_is_per_method_and_seed(tmp_path):
    result, argv = run_train_script(tmp_path, "gpt2-xl", "sra", seed=43)
    assert argv[argv.index("--output_dir") + 1].endswith("gpt2-xl/sra/seed-43") and last_value(argv, "--seed") == "43"
    assert "gpt2-xl/sra/seed-43" in result.stdout


def test_missing_gain_files_stop_before_training_with_a_message(tmp_path):
    cases = [("student", ("teacher",), "gain-tokens-gpt2-xl.json"), ("teacher", ("student",), "gain-tokens-Qwen2.5-7B-Instruct"),
             ("diff", ("student",), "gain-tokens-Qwen2.5-7B-Instruct"), ("diff", ("teacher",), "gain-tokens-gpt2-xl")]
    for source, present, missing in cases:
        result, argv = run_train_script(tmp_path / f"{source}-{'-'.join(present)}", "gpt2-xl", "alerd-ctkd",
                                        extra_env={"GAIN_SOURCE": source}, gain_files=present, check=False)
        assert result.returncode == 3 and "score_gain.sh" in result.stderr and not argv
    result, _ = run_train_script(tmp_path / "pred", "gpt2-xl", "alerd-ctkd-pred-easy", gain_files=(), check=False)
    assert result.returncode == 0  # predictability scores need no gain files
    result, argv = run_train_script(tmp_path / "attention", "gpt2-xl", "sra", gain_files=(), check=False)
    assert result.returncode == 0 and argv


def test_unknown_method_and_student_are_rejected(tmp_path):
    env = dict(os.environ, PYTHON="echo", WORK_DIR=str(tmp_path))
    for student, method in (("nope", "sra"), ("gpt2-xl", "nope")):
        result = subprocess.run([BASH, "scripts/alerd_ctkd_train.sh", student, method], cwd=ROOT, env=env,
                                capture_output=True)
        assert result.returncode == 2


def run_all(tmp_path, student, selection, env=None):
    work = make_work(tmp_path, student)
    (work / STUDENT_DIRS[student] / "spans-Qwen2.5-7B-Instruct.jsonl").write_text("{}\n")
    full_env = dict(os.environ, PYTHON="echo", WORK_DIR=str(work), OUT_ROOT=str(tmp_path / "out"),
                    RESULTS_DIR=str(tmp_path / "results"), **(env or {}))
    result = subprocess.run([BASH, "scripts/alerd_ctkd_run_all.sh", student, *selection], cwd=ROOT, env=full_env,
                            check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if "benchmark_eval.py" in line]


def tag_of(line):
    parts = line.split()
    return parts[parts.index("--tag") + 1]


def test_run_all_main_evaluates_zero_shot_and_teacher_then_every_method_and_seed(tmp_path):
    lines = run_all(tmp_path, "gpt2-xl", ["main"], {"SEEDS": "42 43"})
    tags = [tag_of(line) for line in lines]
    assert tags[:2] == ["gpt2-xl-zeroshot", "teacher"]
    assert tags[2:] == [f"gpt2-xl-{m}-s{s}" for m in ("sft", "sra", "sra-uniform", "alerd-ctkd") for s in (42, 43)]
    gpt2_lines = [line for line in lines if "--tag gpt2-xl-" in line]
    assert all("--max-tokens 512 --max-model-len 1024" in line for line in gpt2_lines)
    assert "--max-tokens 1024 --max-model-len 2048" in lines[1]  # the teacher
    assert all("--benchmarks gsm8k,math500" in line and "--n-samples 3" in line
               for line in lines)
    assert not any("--chat-template" in line for line in lines)


def test_run_all_llama_uses_longer_generation_and_ablations_use_one_seed(tmp_path):
    lines = run_all(tmp_path, "llama-3.2-1b", ["ablations"], {"SEEDS": "42 43"})
    tags = [tag_of(line) for line in lines]
    assert tags[0] == "llama-3.2-1b-alerd-ctkd-gain-student-s42" and len(tags) == 14
    assert all(t.endswith("-s42") for t in tags) and "llama-3.2-1b-alerd-ctkd-unit-step-s42" in tags
    assert all("--max-tokens 1024 --max-model-len 2048" in line for line in lines)
    assert all("--base-model meta-llama/Llama-3.2-1B-Instruct" in line for line in lines)


def test_run_all_rejects_an_unknown_student(tmp_path):
    result = subprocess.run([BASH, "scripts/alerd_ctkd_run_all.sh", "r1-qwen-1.5b"], cwd=ROOT, capture_output=True)
    assert result.returncode == 2


def test_score_gain_script_scores_students_then_the_teacher_once(tmp_path):
    env = dict(os.environ, PYTHON="echo")
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path / "work")], cwd=ROOT, env=env, check=True,
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    commands = [line for line in lines if "token_gain.py records" in line]
    assert len(commands) == 3 and "--side student" in commands[0] and "--side student" in commands[1]
    assert "--side teacher" in commands[2] and "Qwen2.5-7B-Instruct" in commands[2]
    assert sum("answer_gain.py" in line for line in lines) == 3
    assert sum("score_predictability.py" in line for line in lines) == 2  # students only
    assert any("prepare_sra_data.py build" in line and "gpt2-xl" in line for line in lines)


def test_score_gain_word_mode_with_a_limit_scores_a_subset_and_exports_the_span_signals(tmp_path):
    limited = tmp_path / "work-limit5"
    limited.mkdir()
    (limited / "gsm8k-clean.jsonl").write_text("{}\n" * 9)
    env = dict(os.environ, PYTHON="echo", STUDENTS="Qwen/Qwen2.5-1.5B-Instruct", TEACHER="Qwen/Qwen2.5-14B-Instruct",
               SPAN_MODE="word", LIMIT="5")
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path / "work")], cwd=ROOT, env=env, check=True,
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert len((limited / "gsm8k-clean.jsonl").read_text().splitlines()) == 5
    assert sum("answer_gain.py" in line for line in lines) == 2  # the student and the teacher
    assert not any("score_predictability.py" in line for line in lines)
    assert any("span_index.py" in line and "--span-mode word" in line and "spans-Qwen2.5-14B-Instruct.jsonl" in line
               for line in lines)
    assert any("token_gain.py signals" in line and "span-gains.json" in line for line in lines)
    assert sum(line.startswith("scored ") for line in lines) == 2


def test_score_gain_rejects_an_unknown_span_mode(tmp_path):
    result = subprocess.run([BASH, "scripts/score_gain.sh", str(tmp_path)], cwd=ROOT,
                            env=dict(os.environ, PYTHON="echo", SPAN_MODE="phrase"), capture_output=True)
    assert result.returncode == 2
