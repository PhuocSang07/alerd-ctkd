"""Run scripts/score_gain.sh on a Modal GPU, e.g. the cost measurement of the same-family word-span scoring.

    modal run scripts/modal_score_gain.py --limit 1000

Public models and GSM8K only, so no Hugging Face token is needed; weights are cached in the `hf-cache` volume. The script
prints the probe counts and the wall time of each scoring run, then summary statistics of span-gains.json.
"""
import json
import subprocess

import modal

REPO = "/root/repo"
app = modal.App("alerd-ctkd-score-gain")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "datasets", "numpy", "math-verify")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(".", REPO, ignore=["work", "work-*", "outputs", "results", "tests", "plans", "references", ".git",
                                      "**/__pycache__"])
)


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache}, timeout=4 * 3600)
def score(limit: int, student: str, teacher: str, utilization: float) -> dict:
    env = {"STUDENTS": student, "TEACHER": teacher, "SPAN_MODE": "word", "GPU_MEMORY_UTILIZATION": str(utilization),
           "LIMIT": str(limit) if limit else "", "PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/root/.cache/huggingface"}
    result = subprocess.run(["bash", "scripts/score_gain.sh", "work"], cwd=REPO, env=env, text=True)
    cache.commit()
    work = f"{REPO}/work-limit{limit}" if limit else f"{REPO}/work"
    path = f"{work}/{student.split('/')[-1]}/span-gains.json"
    stats = {"returncode": result.returncode}
    if result.returncode == 0:
        gains = json.load(open(path))
        for key in ("student", "teacher", "diff"):
            values = [g for sample in gains.values() for g in sample[key]]
            mean = sum(values) / len(values)
            stats[key] = {"spans": len(values), "mean": mean, "positive": sum(v > 0 for v in values) / len(values),
                          "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5}
    return stats


@app.local_entrypoint()
def main(limit: int = 1000, student: str = "Qwen/Qwen2.5-1.5B-Instruct", teacher: str = "Qwen/Qwen2.5-14B-Instruct",
         utilization: float = 0.85):
    print(json.dumps(score.remote(limit, student, teacher, utilization), indent=2))
