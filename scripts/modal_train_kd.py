"""Train the span-weighted CE + KD student under several weightings in parallel on Modal and evaluate each on GSM8K test.

    modal run --detach scripts/modal_train_kd.py --work work-limit1000 --tag e1k

Inputs are the outputs of scripts/modal_score_gain.py in the `alerd-ctkd-results` volume (train-sra.jsonl, the word-span
index and span-gains.json of <work>). Every experiment of EXPERIMENTS runs on its own A100-80GB (train, then GSM8K test with
benchmark_eval.py); the zero-shot student is evaluated once. Checkpoints stay in the container. Per experiment the volume gets
<work>/train-<tag>/<name>/{train_stats.jsonl, spans.jsonl.gz, eval/, result.json}, and the entrypoint prints one table.
"""
import json
import subprocess
import time

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

# name: (CE masses, KD masses, beta, ALERD lambda); the gains enter the two component losses separately
EXPERIMENTS = {
    "sft": ("token", "token", 0.0, 0.5),               # CE only, the teacher is not loaded
    "kd-token": ("token", "token", 0.5, 0.5),          # plain token-mean CE + KD
    "kd-uniform": ("uniform", "uniform", 0.5, 0.5),    # every span the same mass
    "gain-ce": ("student", "token", 0.5, 0.5),         # student gain on the CE only
    "gain-kd": ("token", "diff", 0.5, 0.5),            # teacher - student gain on the KD only
    "gain-both": ("student", "diff", 0.5, 0.5),        # the proposed pairing
    "gain-student": ("student", "student", 0.5, 0.5),  # student gain on both
    "gain-diff": ("diff", "diff", 0.5, 0.5),           # teacher - student gain on both
    "gain-both-lam1": ("student", "diff", 0.5, 1.0),   # the full gain weights, no uniform share
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, cwd=REPO, env=env, check=True)


def evaluate(model: str, name: str, student: str, n_samples: int, out: str) -> dict:
    run(["python", "benchmark_eval.py", "--model", model, "--tag", name, "--base-model", student,
         "--benchmarks", "gsm8k", "--n-samples", str(n_samples), "--results-dir", f"{out}/eval",
         "--gpu-memory-utilization", "0.5"])
    return json.load(open(f"{out}/eval/{name}/summary.json"))


def keep(out: str, work: str, tag: str, name: str, result: dict) -> None:
    folder = f"/root/results/{work}/train-{tag}/{name}"
    subprocess.run(f"mkdir -p {folder} && cp -r {out}/eval {folder}/", shell=True, check=True)
    subprocess.run(f"cp {out}/train_stats.jsonl {folder}/ && gzip -c {out}/spans.jsonl > {folder}/spans.jsonl.gz",
                   shell=True, check=False)
    json.dump(result, open(f"{folder}/result.json", "w"), indent=2)
    cache.commit()
    results.commit()


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache, "/root/results": results},
              timeout=6 * 3600)
def experiment(work: str, student: str, teacher: str, tag: str, name: str, epochs: int, batch_size: int,
               micro_batch_size: int, kd_top_k: int, learning_rate: float, eval_n_samples: int) -> dict:
    ce, kd, beta, lam = EXPERIMENTS[name]
    data = f"/root/results/{work}/{student.split('/')[-1]}"
    out = f"/root/run/{name}"
    started = time.time()
    command = ["python", "span_kd_train.py", "--train-data", f"{data}/train-sra.jsonl",
               "--spans-file", f"{data}/spans-{teacher.split('/')[-1]}.jsonl", "--gains-file", f"{data}/span-gains.json",
               "--student-model", student, "--output-dir", out, "--ce-weights", ce, "--kd-weights", kd,
               "--kd-beta", str(beta), "--kd-top-k", str(kd_top_k), "--alloc-lambda", str(lam),
               "--epochs", str(epochs), "--batch-size", str(batch_size), "--micro-batch-size", str(micro_batch_size),
               "--learning-rate", str(learning_rate), "--dump-spans", f"{out}/spans.jsonl"]
    if beta > 0:
        command += ["--teacher-model", teacher]
    run(command)
    train_seconds = round(time.time() - started)  # model loading and the first download included
    rows = [json.loads(line) for line in open(f"{out}/train_stats.jsonl")]
    result = {"name": name, "ce_weights": ce, "kd_weights": kd, "beta": beta, "alloc_lambda": lam,
              "train_seconds": train_seconds, "steps": len(rows), "first": rows[0], "last": rows[-1],
              "gsm8k": evaluate(out, name, student, eval_n_samples, out)}
    keep(out, work, tag, name, result)
    return result


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache, "/root/results": results},
              timeout=2 * 3600)
def zero_shot(work: str, student: str, tag: str, eval_n_samples: int) -> dict:
    out = "/root/run/zero-shot"
    result = {"name": "zero-shot", "gsm8k": evaluate(student, "zero-shot", student, eval_n_samples, out)}
    folder = f"/root/results/{work}/train-{tag}/zero-shot"
    subprocess.run(f"mkdir -p {folder} && cp -r {out}/eval {folder}/", shell=True, check=True)
    json.dump(result, open(f"{folder}/result.json", "w"), indent=2)
    results.commit()
    return result


@app.local_entrypoint()
def main(work: str = "work-limit1000", student: str = "Qwen/Qwen2.5-1.5B-Instruct",
         teacher: str = "Qwen/Qwen2.5-14B-Instruct", tag: str = "e1k", names: str = "", epochs: int = 3,
         batch_size: int = 32, micro_batch_size: int = 8, kd_top_k: int = 50, learning_rate: float = 1e-4,
         eval_n_samples: int = 3):
    chosen = [n for n in (names.split(",") if names else EXPERIMENTS) if n]
    calls = [experiment.spawn(work, student, teacher, tag, n, epochs, batch_size, micro_batch_size, kd_top_k,
                              learning_rate, eval_n_samples) for n in chosen]
    calls.append(zero_shot.spawn(work, student, tag, eval_n_samples))
    print(f"{'name':16} {'pass@1':>7} {'pass@3':>7} {'ce_tok':>7} {'kd_tok':>7} {'secs':>6}")
    for call in calls:
        r = call.get()
        g = r["gsm8k"]  # benchmark_eval summary: a list with one row per benchmark
        last = r.get("last", {})
        print(f"{r['name']:16} {g[0]['pass@1']:7.4f} {g[0].get('pass@3', 0):7.4f} {last.get('ce_token', 0):7.3f} "
              f"{last.get('kd_token', 0):7.3f} {r.get('train_seconds', 0):6}", flush=True)
