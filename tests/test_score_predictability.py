import json
import os
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_predictability import main, step_mean_logprobs  # noqa: E402


def tiny_model(tmp_path):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(transformers.Qwen2Config(
        vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1)).eval()
    model.save_pretrained(tmp_path / "m")
    return model


def test_step_mean_logprobs_scores_each_token_given_its_prefix(tmp_path):
    model = tiny_model(tmp_path)
    ids = [3, 8, 9, 4, 7, 1, 2, 6]
    record = {"id": 5, "input_ids": ids, "steps": [{"token_start": 3, "token_end": 5}, {"token_start": 5, "token_end": 8}]}
    got = step_mean_logprobs(model, record)
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids])).logits[0], dim=-1)
    expect = [sum(logp[t - 1, ids[t]].item() for t in range(a, b)) / (b - a) for a, b in ((3, 5), (5, 8))]
    assert all(abs(g - e) < 1e-5 for g, e in zip(got, expect))


def test_cli_writes_a_score_per_step_keyed_by_id(tmp_path):
    tiny_model(tmp_path)
    records = [{"id": i, "input_ids": list(range(1, 11)), "steps": [{"token_start": 4, "token_end": 7},
                                                                    {"token_start": 7, "token_end": 10}]}
               for i in (2, 9)]
    data = tmp_path / "seg.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records))
    main(["--data", str(data), "--model", str(tmp_path / "m"), "--output", str(tmp_path / "p.json"),
          "--dtype", "float32"])
    out = json.loads((tmp_path / "p.json").read_text())
    assert set(out) == {"2", "9"} and all(len(v) == 2 and all(x < 0 for x in v) for v in out.values())
