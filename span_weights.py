"""Per-sample span weights omega_n = w_k(n) / N from step scores, for the collator.

Inputs are the offline artefacts of phases 1 and 3: spans.jsonl (span_index.py), the answer-gain json of
sgl.signals.answer_gain ({id: [gain per step]}) and the SRA training jsonl. A sample whose text hash or span
count differs from the index stops the run: weights computed for other spans would be silently wrong.
"""
import json
import random
from collections import Counter

import torch

from allocation import (corpus_weights, last_boxed, span_weights, step_weights_for_source,
                        trace_weights)

SOURCES = ("uniform", "alerd", "random", "predictability-easy", "predictability-hard")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class SpanWeightProvider:
    """omega lists per example id, built once from the score files."""

    def __init__(self, source, spans_file, train_data, gain_file=None, predictability_file=None, tau=1.0, clip=2.0,
                 lam=0.5, normalization="per-trace", answer_only_unit=False, seed=0):
        if source not in SOURCES:
            raise ValueError(f"span weight source must be one of {SOURCES}, got {source!r}")
        if source == "random" and gain_file is None:
            raise ValueError("source 'random' permutes the gains and needs gain_file")
        if source.startswith("predictability") and predictability_file is None:
            raise ValueError(f"source {source!r} needs predictability_file")
        if source == "alerd" and gain_file is None:
            raise ValueError("source 'alerd' needs gain_file")
        if normalization == "corpus" and source != "alerd":
            raise ValueError("corpus normalization is defined for the alerd source only")
        if normalization == "corpus" and answer_only_unit:
            raise ValueError("corpus normalization cannot be combined with answer_only_unit")

        self.spans = {row["id"]: row for row in load_jsonl(spans_file)}
        self.omega = {}
        gains = self._read_scores(gain_file)
        logprobs = self._read_scores(predictability_file)
        rows = {row["id"]: row for row in load_jsonl(train_data)} if answer_only_unit else {}

        traces = []  # (id, scores, masses) for every indexed sample
        for rid, span_row in self.spans.items():
            if span_row["step_of_span"] is None:
                raise ValueError(f"spans.jsonl has no step_of_span for id {rid}; build it from data with step_char_ends")
            n_steps = span_row["n_steps"]
            masses = [0] * n_steps
            for k, count in Counter(span_row["step_of_span"]).items():
                masses[k] = count
            traces.append((rid, masses))

        if source == "uniform":
            for rid, span_row in self.spans.items():
                n = len(span_row["step_of_span"])
                self.omega[rid] = [1.0 / n] * n
            return

        step_w = {}
        if source == "alerd" and normalization == "corpus" and not answer_only_unit:
            inputs = [(self._scores(gains, rid, m), m) for rid, m in traces]
            for (rid, _), weights in zip(traces, corpus_weights(inputs, tau, lam, clip, 1e-8, "corpus")):
                step_w[rid] = weights
        else:
            for rid, masses in traces:
                step_w[rid] = self._step_weights(source, rid, masses, gains, logprobs, rows, tau, clip, lam,
                                                 normalization, answer_only_unit, seed)
        for rid, weights in step_w.items():
            self.omega[rid] = span_weights(weights, self.spans[rid]["step_of_span"])

    @staticmethod
    def _read_scores(path):
        if path is None:
            return None
        with open(path, encoding="utf-8") as handle:
            return {int(k): v for k, v in json.load(handle).items()}

    def _scores(self, table, rid, masses):
        if rid not in table:
            raise ValueError(f"no scores for id {rid}")
        if len(table[rid]) != len(masses):
            raise ValueError(f"id {rid}: {len(table[rid])} scores for {len(masses)} steps")
        return table[rid]

    def _step_weights(self, source, rid, masses, gains, logprobs, rows, tau, clip, lam, normalization,
                      answer_only_unit, seed):
        kwargs = dict(masses=masses, tau=tau, clip=clip, lam=lam, normalization=normalization)
        if source == "alerd":
            scores = self._scores(gains, rid, masses)
            if answer_only_unit:
                restates = self._restates_answer(rows[rid])
                return step_weights_for_source("answer-only-unit", scores, restates_answer=restates, **kwargs)
            return step_weights_for_source("gain", scores, **kwargs)
        if source == "random":
            return step_weights_for_source("random", self._scores(gains, rid, masses),
                                           seed=random.Random(f"{seed}-{rid}").randrange(2 ** 31), **kwargs)
        return step_weights_for_source(source, [0.0] * len(masses),
                                       mean_logprobs=self._scores(logprobs, rid, masses), **kwargs)

    @staticmethod
    def _restates_answer(row):
        """Steps whose text contains the gold answer (the last boxed answer of the solution)."""
        answer = last_boxed(row["output"])
        ends = row["step_char_ends"]
        starts = [0] + ends[:-1]
        return [bool(answer) and answer in row["output"][a:b] for a, b in zip(starts, ends)]

    def check(self, example_id, text_hash, n_spans):
        row = self.spans.get(example_id)
        if row is None:
            raise ValueError(f"id {example_id} is not in the span index")
        if row["hash"] != text_hash:
            raise ValueError(f"id {example_id}: text differs from the one indexed (hash mismatch); rebuild spans.jsonl")
        if len(row["step_of_span"]) != n_spans:
            raise ValueError(f"id {example_id}: {n_spans} spans at train time, {len(row['step_of_span'])} indexed")

    def batch_weights(self, example_ids, hashes, n_spans, width):
        """(B, width) tensor of omega, zero-padded; stops on any hash or span-count mismatch."""
        out = torch.zeros(len(example_ids), width)
        for b, (rid, text_hash, n) in enumerate(zip(example_ids, hashes, n_spans)):
            self.check(rid, text_hash, n)
            out[b, :n] = torch.tensor(self.omega[rid])
        return out
