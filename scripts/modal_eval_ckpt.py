"""Evaluate the checkpoints saved by scripts/modal_train_kd.py with a long generation budget (4096 new tokens).

    modal run --detach scripts/modal_eval_ckpt.py --tag e1k-ckpt

Per checkpoint (and the zero-shot student) one A100-80GB: MATH500 with 3 samples per problem, AIME24 and AIME25 with 16
(pass@1 over all samples, pass@3 estimated without bias, pass@16 = any correct), temperature 0.6, top-p 0.9. Raw
generations and summaries go to <work>/eval<max_tokens>-<tag>/<name>/ in the `alerd-ctkd-results` volume.
"""
import json
import subprocess

import modal

REPO = "/root/repo"
app = modal.App("alerd-ctkd-eval-ckpt")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("alerd-ctkd-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "datasets", "numpy", "math-verify")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(".", REPO, ignore=["work", "work-*", "outputs", "results", "tests", "plans", "references", ".git",
                                      "**/__pycache__"])
)
NAMES = ["sft", "kd-token", "kd-uniform", "gain-ce", "gain-kd", "gain-both", "gain-student", "gain-diff",
         "gain-both-lam1"]
SUITES = (("math500", 3), ("aime24,aime25", 16))
BENCHMARKS = ("math500", "aime24", "aime25")


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache, "/root/results": results},
              timeout=4 * 3600)
def evaluate(work: str, tag: str, name: str, student: str, max_tokens: int) -> dict:
    model = student if name == "zero-shot" else f"/root/results/{work}/ckpt-{tag}/{name}"
    out = f"/root/results/{work}/eval{max_tokens}-{tag}/{name}"
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"}
    rows = []
    for benchmarks, n in SUITES:
        suite = benchmarks.split(",")[0]
        subprocess.run(["python", "benchmark_eval.py", "--model", model, "--tag", suite, "--base-model", student,
                        "--benchmarks", benchmarks, "--n-samples", str(n), "--max-tokens", str(max_tokens),
                        "--max-model-len", str(max_tokens + 2048), "--gpu-memory-utilization", "0.85",
                        "--results-dir", out], cwd=REPO, env=env, check=True)
        rows += json.load(open(f"{out}/{suite}/summary.json"))
    json.dump(rows, open(f"{out}/result.json", "w"), indent=2)
    cache.commit()
    results.commit()
    return {"name": name, "rows": rows}


@app.local_entrypoint()
def main(work: str = "work-limit1000", tag: str = "e1k-ckpt", student: str = "Qwen/Qwen2.5-1.5B-Instruct",
         max_tokens: int = 4096, names: str = ""):
    chosen = [n for n in (names.split(",") if names else ["zero-shot"] + NAMES) if n]
    calls = [evaluate.spawn(work, tag, n, student, max_tokens) for n in chosen]
    print(f"{'name':15} " + " ".join(f"{b + ' p@1':>12} {'p@3':>6} {'len':>6} {'trunc%':>6}" for b in BENCHMARKS))
    for call in calls:
        r = call.get()
        by = {row["benchmark"]: row for row in r["rows"]}
        print(f"{r['name']:15} " + " ".join(
            f"{100 * by[b]['pass@1']:12.2f} {100 * by[b]['pass@3']:6.2f} {by[b]['length']:6.0f} "
            f"{100 * by[b]['truncation_rate']:6.2f}" for b in BENCHMARKS), flush=True)
