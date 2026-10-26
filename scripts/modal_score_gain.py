"""Run scripts/score_gain.sh on Modal GPUs, e.g. the word-span gain scoring of GSM8K or s1K.

    modal run --detach scripts/modal_score_gain.py --limit 1000                    # GSM8K, the first 1,000 problems
    modal run --detach scripts/modal_score_gain.py --dataset s1k --shards 6        # s1K-1.1, six GPUs in parallel

Public models and datasets only, so no Hugging Face token is needed; weights are cached in the `hf-cache` volume. The
script prints the probe counts and the wall time of each scoring run, then summary statistics of span-gains.json. With
--shards N the rows are dealt to N containers (SHARD=i/N of score_gain.sh) and merged afterwards: the merge writes the
training file, the word-span index and span-gains.json to `<work>/<student>/` of the `alerd-ctkd-results` volume, where
scripts/modal_train_kd.py reads them (<work> is `work`, `work-limit<N>` or `work-<dataset>`).
"""
import json
import subprocess

import modal

REPO = "/root/repo"
app = modal.App("alerd-ctkd-score-gain")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("alerd-ctkd-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "datasets", "numpy", "math-verify")
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(".", REPO, ignore=["work", "work-*", "outputs", "results", "tests", "plans", "references", ".git",
                                      "**/__pycache__"])
)
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"}


def base_name(dataset: str, limit: int) -> str:
    return ("work" if dataset == "gsm8k" else f"work-{dataset}") + (f"-limit{limit}" if limit else "")


def gain_stats(path: str) -> dict:
    gains = json.load(open(path))
    stats = {}
    for key in ("student", "teacher", "diff"):
        values = [g for sample in gains.values() for g in sample[key]]
        mean = sum(values) / len(values)
        stats[key] = {"spans": len(values), "mean": mean, "positive": sum(v > 0 for v in values) / len(values),
                      "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5}
    return stats


@app.function(image=image, gpu="A100-80GB", volumes={"/root/.cache/huggingface": cache, "/root/results": results},
              timeout=6 * 3600)
def score(limit: int, student: str, teacher: str, utilization: float, dataset: str = "gsm8k", shard: str = "") -> dict:
    env = dict(ENV, STUDENTS=student, TEACHER=teacher, SPAN_MODE="word", GPU_MEMORY_UTILIZATION=str(utilization),
               LIMIT=str(limit) if limit else "", DATASET=dataset, SHARD=shard)
    subprocess.run(r"sed -i 's/\r$//' scripts/*.sh", shell=True, cwd=REPO, check=True)  # sources checked out on Windows
    root = "work" if dataset == "gsm8k" else f"work-{dataset}"  # score_gain.sh appends -limit<N> and -shard<i>of<n>
    result = subprocess.run(["bash", "scripts/score_gain.sh", root], cwd=REPO, env=env, text=True)
    work = f"{REPO}/{base_name(dataset, limit)}" + (f"-shard{shard.split('/')[0]}of{shard.split('/')[1]}" if shard else "")
    stats = {"returncode": result.returncode, "shard": shard}
    if result.returncode == 0 and not shard:
        stats.update(gain_stats(f"{work}/{student.split('/')[-1]}/span-gains.json"))
    subprocess.run(f"cp -r {work} /root/results/", shell=True, check=False)
    cache.commit()
    results.commit()
    return stats


@app.function(image=image, volumes={"/root/.cache/huggingface": cache, "/root/results": results}, cpu=4,
              timeout=2 * 3600)
def merge(dataset: str, limit: int, shards: int, student: str, teacher: str, max_len: int) -> dict:
    """Join the shards' training rows and gain files, then build the span index and the span gains."""
    results.reload()
    base = base_name(dataset, limit)
    s_tag, t_tag = student.split("/")[-1], teacher.split("/")[-1]
    shard_dirs = [f"/root/results/{base}-shard{i}of{shards}" for i in range(shards)]
    out, t_out = f"/root/results/{base}/{s_tag}", f"/root/results/{base}/{t_tag}"
    subprocess.run(["mkdir", "-p", out, t_out], check=True)
    rows = []
    for directory in shard_dirs:
        rows += [json.loads(line) for line in open(f"{directory}/{s_tag}/train-sra.jsonl", encoding="utf-8")]
    rows.sort(key=lambda row: row["id"])
    with open(f"{out}/train-sra.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(command: list) -> None:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, env=ENV, check=True)

    run(["python", "token_gain.py", "merge", "--inputs", *[f"{d}/{s_tag}/gain-tokens-{s_tag}.json" for d in shard_dirs],
         "--output", f"{out}/gain-tokens-{s_tag}.json"])
    run(["python", "token_gain.py", "merge", "--inputs", *[f"{d}/{t_tag}/gain-tokens-{t_tag}.json" for d in shard_dirs],
         "--output", f"{t_out}/gain-tokens-{t_tag}.json"])
    run(["python", "span_index.py", "--data", f"{out}/train-sra.jsonl", "--student-tokenizer", student,
         "--teacher-tokenizer", teacher, "--span-mode", "word", "--max-len", str(max_len),
         "--output", f"{out}/spans-{t_tag}.jsonl"])
    run(["python", "token_gain.py", "signals", "--spans", f"{out}/spans-{t_tag}.jsonl",
         "--student-gains", f"{out}/gain-tokens-{s_tag}.json", "--teacher-gains", f"{t_out}/gain-tokens-{t_tag}.json",
         "--output", f"{out}/span-gains.json"])
    results.commit()
    return {"samples": len(rows), **gain_stats(f"{out}/span-gains.json")}


@app.function(image=image, volumes={"/root/results": results}, timeout=8 * 3600)
def run_sharded(limit: int, student: str, teacher: str, utilization: float, dataset: str, shards: int,
                max_len: int) -> dict:
    """Scores the shards on parallel GPUs and merges them; running remotely, it outlives the local client."""
    calls = [score.spawn(limit, student, teacher, utilization, dataset, f"{i}/{shards}") for i in range(shards)]
    reports = [call.get() for call in calls]
    if any(report["returncode"] for report in reports):
        return {"failed": reports}
    summary = {"shards": reports, "merged": merge.remote(dataset, limit, shards, student, teacher, max_len)}
    path = f"/root/results/{base_name(dataset, limit)}/sharded-run.json"
    json.dump(summary, open(path, "w"), indent=2)
    results.commit()
    return summary


@app.local_entrypoint()
def main(limit: int = 0, student: str = "Qwen/Qwen2.5-1.5B-Instruct", teacher: str = "Qwen/Qwen2.5-14B-Instruct",
         utilization: float = 0.85, dataset: str = "gsm8k", shards: int = 1, max_len: int = 0):
    if shards <= 1:
        print(json.dumps(score.remote(limit, student, teacher, utilization, dataset), indent=2))
        return
    call = run_sharded.spawn(limit, student, teacher, utilization, dataset, shards,
                             max_len or (1024 if dataset == "s1k" else 512))
    print(f"started {call.object_id}; the result lands in {base_name(dataset, limit)}/sharded-run.json of the "
          "alerd-ctkd-results volume (run with --detach, the client may exit)")
