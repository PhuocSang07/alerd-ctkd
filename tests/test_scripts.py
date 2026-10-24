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

METHODS = ["sra", "sra-uniform", "alerd-ctkd", "alerd-ctkd-random", "alerd-ctkd-pred-easy", "alerd-ctkd-pred-hard",
           "alerd-ctkd-answer-only-unit", "alerd-ctkd-corpus", "alerd-ctkd-nonorm", "alerd-ctkd-t2-l0",
           "alerd-ctkd-t2-l05", "alerd-ctkd-t2-l1", "alerd-ctkd-t4-l05"]


def last_value(argv, flag):
    """argparse keeps the last occurrence of a repeated flag."""
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


def run_train_script(tmp_path, student, method):
    """Run alerd_ctkd_train.sh with `echo` standing in for python; returns the arguments it would pass."""
    work, out = tmp_path / "work", tmp_path / "out"
    model_dir = work / {"r1-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B", "qwen25-7b": "Qwen2.5-7B-Instruct",
                        "qwen3-8b": "Qwen3-8B"}[student]
    model_dir.mkdir(parents=True)
    (model_dir / "train-sra.jsonl").write_text("{}\n")
    env = dict(os.environ, PYTHON="echo", TEACHER="some/teacher", TEACHER_LAYER="30", TEACHER_DIM="4096",
               WORK_DIR=str(work), OUT_ROOT=str(out))
    subprocess.run([BASH, "scripts/alerd_ctkd_train.sh", student, method], cwd=ROOT, env=env, check=True,
                   capture_output=True)
    log = (out / student / method / "train.log").read_text()
    return log.replace("run_distill_llm.py", "").split()


@pytest.mark.parametrize("method", METHODS)
def test_every_method_yields_arguments_the_trainer_accepts(tmp_path, method):
    argv = run_train_script(tmp_path, "r1-qwen-1.5b", method)
    # the data files are not there (echo stands in for the scripts), so only parse; do not build the Arguments
    train_data = argv[argv.index("--train_data") + 1]
    argv[argv.index("--train_data") + 1] = str(tmp_path / "train.jsonl")
    (tmp_path / "train.jsonl").write_text("{}\n")
    args, extras = parse(argv)
    assert isinstance(args, Arguments) and train_data.endswith("train-sra.jsonl")
    assert args.num_train_epochs == 3 and args.grad_accum_steps == 32 and args.batch_size == 1
    assert args.learning_rate == 5e-5 and args.min_learning_rate == 1e-5
    assert args.student_encoder_layers_finetuned == [28] and args.teacher_layers_mapping == [30]
    assert args.use_lora is False and args.student_dtype == "float32"


@pytest.mark.parametrize("method,source,tau,lam", [
    ("sra", "attention", 1.0, 0.5), ("sra-uniform", "uniform", 1.0, 0.5), ("alerd-ctkd", "alerd", 1.0, 0.5),
    ("alerd-ctkd-random", "random", 1.0, 0.5), ("alerd-ctkd-pred-easy", "predictability-easy", 1.0, 0.5),
    ("alerd-ctkd-pred-hard", "predictability-hard", 1.0, 0.5), ("alerd-ctkd-t2-l0", "alerd", 2.0, 0.0),
    ("alerd-ctkd-t2-l05", "alerd", 2.0, 0.5), ("alerd-ctkd-t2-l1", "alerd", 2.0, 1.0),
    ("alerd-ctkd-t4-l05", "alerd", 4.0, 0.5)])
def test_method_selects_the_expected_weights(tmp_path, method, source, tau, lam):
    argv = run_train_script(tmp_path, "r1-qwen-1.5b", method)
    assert argv[argv.index("--span_weight_source") + 1] == source
    if source != "attention" and source != "uniform":
        assert float(last_value(argv, "--alloc_tau")) == tau
        assert float(last_value(argv, "--alloc_lambda")) == lam


def test_ablation_flags_and_lora_students(tmp_path):
    argv = run_train_script(tmp_path, "qwen3-8b", "alerd-ctkd-corpus")
    norms = [argv[i + 1] for i, a in enumerate(argv) if a == "--alloc_norm"]
    assert norms == ["per-trace", "corpus"]  # the method's flag comes last and overrides the default
    assert argv[argv.index("--use_lora") + 1] == "True" and argv[argv.index("--student_dtype") + 1] == "bfloat16"
    assert argv[argv.index("--student_encoder_layers_finetuned") + 1] == "36"
    assert "--answer_only_unit" in run_train_script(tmp_path / "b", "qwen25-7b", "alerd-ctkd-answer-only-unit")


def test_unknown_method_and_student_are_rejected(tmp_path):
    env = dict(os.environ, PYTHON="echo", TEACHER="t", TEACHER_LAYER="1", TEACHER_DIM="8", WORK_DIR=str(tmp_path))
    for student, method in (("nope", "sra"), ("r1-qwen-1.5b", "nope")):
        result = subprocess.run([BASH, "scripts/alerd_ctkd_train.sh", student, method], cwd=ROOT, env=env,
                                capture_output=True)
        assert result.returncode == 2
