"""Train the span-weighted CE + KD student on a Modal GPU and evaluate it on GSM8K test, next to the plain token-weighted run.

    modal run --detach scripts/modal_train_kd.py --work work-limit200 --epochs 3 --batch-size 16

Inputs are the outputs of scripts/modal_score_gain.py in the `alerd-ctkd-results` volume (train-sra.jsonl, the word-span
index and span-gains.json of <work>). Runs, in one container: the token-weighted baseline, the gain-weighted run (student
gain for the CE masses, teacher minus student for the KD masses), then the zero-shot student and both trained students on
GSM8K test (benchmark_eval.py). Checkpoints stay in the container; stats and summaries are copied to the volume under
<work>/train-<tag>/ and returned.
"""
import json
import subprocess

import modal

REPO = "/root/repo"
app = modal.App("alerd-ctkd-train-kd")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("alerd-ctkd-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft", "datasets", "numpy", "math-verify", "scipy")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(".", REPO, ignore=["work", "work-*", "outputs", "results", "tests", "plans", "references", ".git",
                                      "**/__pycache__"])
)


def run(command: list[str], env: dict) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache, "/root/results": results},
              timeout=4 * 3600)
def train(work: str, student: str, teacher: str, tag: str, epochs: int, batch_size: int, micro_batch_size: int,
          kd_beta: float, kd_top_k: int, learning_rate: float, eval_n_samples: int) -> dict:
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"}
    name = student.split("/")[-1]
    data = f"/root/results/{work}/{name}"
    out = f"/root/run/{tag}"
    runs = {"token": ("token", "token"), "gain": ("student", "diff")}
    for label, (ce, kd) in runs.items():
        run(["python", "span_kd_train.py", "--train-data", f"{data}/train-sra.jsonl",
             "--spans-file", f"{data}/spans-{teacher.split('/')[-1]}.jsonl", "--gains-file", f"{data}/span-gains.json",
             "--student-model", student, "--teacher-model", teacher, "--output-dir", f"{out}/{label}",
             "--ce-weights", ce, "--kd-weights", kd, "--kd-beta", str(kd_beta), "--kd-top-k", str(kd_top_k),
             "--epochs", str(epochs), "--batch-size", str(batch_size), "--micro-batch-size", str(micro_batch_size),
             "--learning-rate", str(learning_rate)], env)
    summary = {}
    for label, model in (("zero-shot", student), *((label, f"{out}/{label}") for label in runs)):
        run(["python", "benchmark_eval.py", "--model", model, "--tag", label, "--base-model", student,
             "--benchmarks", "gsm8k", "--n-samples", str(eval_n_samples), "--results-dir", f"{out}/eval",
             "--gpu-memory-utilization", "0.5"], env)
        summary[label] = json.load(open(f"{out}/eval/{label}/summary.json"))
    for label in runs:
        rows = [json.loads(line) for line in open(f"{out}/{label}/train_stats.jsonl")]
        summary[f"{label}-train"] = {"first": rows[0], "last": rows[-1], "steps": len(rows)}
    keep = f"/root/results/{work}/train-{tag}"
    subprocess.run(f"mkdir -p {keep} && cp -r {out}/eval {keep}/ && "
                   + " && ".join(f"cp {out}/{label}/train_stats.jsonl {keep}/train_stats-{label}.jsonl" for label in runs),
                   shell=True, check=True)
    json.dump(summary, open(f"{keep}/result.json", "w"), indent=2)
    cache.commit()
    results.commit()
    return summary


@app.local_entrypoint()
def main(work: str = "work-limit200", student: str = "Qwen/Qwen2.5-1.5B-Instruct",
         teacher: str = "Qwen/Qwen2.5-14B-Instruct", tag: str = "pilot", epochs: int = 3, batch_size: int = 16,
         micro_batch_size: int = 8, kd_beta: float = 0.5, kd_top_k: int = 50, learning_rate: float = 1e-4,
         eval_n_samples: int = 3):
    print(json.dumps(train.remote(work, student, teacher, tag, epochs, batch_size, micro_batch_size, kd_beta, kd_top_k,
                                  learning_rate, eval_n_samples), indent=2))
