r"""Per-sample span weights omega_n for the collator, from answer-gain scores and the span index.

Inputs are offline artefacts: spans.jsonl (span_index.py), the token-level gain files of the scoring models
(token_gain.py, one for the student and one for the teacher) and the SRA training jsonl.

Sources (`source`):
  uniform                           omega_n = 1/N
  alerd                             answer-gain scores through ALERD Eq. 3-5
  random                            the same scores permuted within the sample
  predictability-easy / -hard       mean token log-probability of each step under the frozen student, per step
Gain scores come from `gain_source` student | teacher | diff (teacher - student, on raw span gains) and are taken per
span (`gain_unit` span: every span is its own unit, mass 1) or per step (spans inherit the gain of their step, a
step weighs as many spans as it holds). Either way sum_n omega_n = 1. A sample whose text hash or span count differs from
the index stops the run: weights computed for other spans would be silently wrong.
"""
import json
import random
from collections import Counter

import torch

from allocation import (contains_answer, corpus_weights, last_boxed, span_weights, step_weights_for_source)
from token_gain import GainFile, span_gains

SOURCES = ("uniform", "alerd", "random", "predictability-easy", "predictability-hard")
GAIN_SOURCES = ("student", "teacher", "diff")
GAIN_UNITS = ("span", "step")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class SpanWeightProvider:
    """omega lists per example id, built once from the score files."""

    def __init__(self, source, spans_file, train_data, student_gain_file=None, teacher_gain_file=None,
                 gain_source="student", gain_unit="span", predictability_file=None, tau=1.0, clip=2.0, lam=0.5,
                 normalization="per-trace", answer_only_unit=False, seed=0):
        if source not in SOURCES:
            raise ValueError(f"span weight source must be one of {SOURCES}, got {source!r}")
        if gain_source not in GAIN_SOURCES:
            raise ValueError(f"gain source must be one of {GAIN_SOURCES}, got {gain_source!r}")
        if gain_unit not in GAIN_UNITS:
            raise ValueError(f"gain unit must be one of {GAIN_UNITS}, got {gain_unit!r}")
        uses_gain = source in ("alerd", "random")
        if uses_gain:
            self.gain_files = {}
            for role, path in (("student", student_gain_file), ("teacher", teacher_gain_file)):
                if gain_source in (role, "diff"):
                    if path is None:
                        raise ValueError(f"gain source {gain_source!r} needs the {role} gain file "
                                         f"(--{role}_gain_file, from token_gain.py pack); none was given")
                    self.gain_files[role] = GainFile(path, role)
        if source.startswith("predictability") and predictability_file is None:
            raise ValueError(f"source {source!r} needs predictability_file")
        if normalization == "corpus" and source != "alerd":
            raise ValueError("corpus normalization is defined for the alerd source only")
        if normalization == "corpus" and answer_only_unit:
            raise ValueError("corpus normalization cannot be combined with answer_only_unit")
        if answer_only_unit and source != "alerd":
            raise ValueError("answer_only_unit applies to the alerd source only")

        self.spans = {row["id"]: row for row in load_jsonl(spans_file)}
        self.omega = {}
        self.gain_source, self.gain_unit = gain_source, gain_unit
        logprobs = self._read_scores(predictability_file)
        rows = {row["id"]: row for row in load_jsonl(train_data)} if answer_only_unit else {}

        for rid, span_row in self.spans.items():
            if span_row["step_of_span"] is None:
                raise ValueError(f"spans.jsonl has no step_of_span for id {rid}; build it from data with step_char_ends")

        if source == "uniform":
            for rid, span_row in self.spans.items():
                n = len(span_row["step_of_span"])
                self.omega[rid] = [1.0 / n] * n
            return

        # one (scores, masses, step_of_unit, restates) bundle per sample; weights are computed per unit
        bundles = {}
        for rid, span_row in self.spans.items():
            if source.startswith("predictability"):
                scores, masses, unit_steps = self._step_unit(span_row, self._step_scores(logprobs, rid, span_row))
            else:
                scores, masses, unit_steps = self._gain_unit(rid, span_row)
            restates = self._restates_answer(rows[rid]) if answer_only_unit else None
            bundles[rid] = (scores, masses, unit_steps, restates)

        if source == "alerd" and normalization == "corpus":
            inputs = [(scores, masses) for scores, masses, _, _ in bundles.values()]
            unit_w = dict(zip(bundles, corpus_weights(inputs, tau, lam, clip, 1e-8, "corpus")))
        else:
            unit_w = {rid: self._unit_weights(source, rid, bundle, tau, clip, lam, normalization, seed)
                      for rid, bundle in bundles.items()}
        for rid, weights in unit_w.items():
            self.omega[rid] = span_weights(weights, bundles[rid][2])

    # -- scores per unit ---------------------------------------------------------------------------------------

    @staticmethod
    def _read_scores(path):
        if path is None:
            return None
        with open(path, encoding="utf-8") as handle:
            return {int(k): v for k, v in json.load(handle).items()}

    @staticmethod
    def _step_masses(span_row):
        masses = [0] * span_row["n_steps"]
        for k, count in Counter(span_row["step_of_span"]).items():
            masses[k] = count
        return masses

    def _step_unit(self, span_row, step_scores):
        """Units are steps: spans map to their step through step_of_span, a step weighs its span count."""
        masses = self._step_masses(span_row)
        if len(step_scores) != len(masses):
            raise ValueError(f"id {span_row['id']}: {len(step_scores)} step scores for {len(masses)} steps")
        return list(step_scores), masses, span_row["step_of_span"]

    @staticmethod
    def _step_scores(table, rid, span_row):
        if rid not in table:
            raise ValueError(f"no predictability scores for id {rid}")
        return table[rid]

    def _entry(self, role, rid, span_row):
        tokens = span_row[f"{role}_tokens"]
        expected_hash = span_row["hash"] if role == "student" else span_row["teacher_hash"]
        return self.gain_files[role].entry(rid, expected_hash, tokens[0][0], tokens[-1][1]), tokens

    def _gain_unit(self, rid, span_row):
        roles = ("student", "teacher") if self.gain_source == "diff" else (self.gain_source,)
        per_role = {}
        for role in roles:
            entry, tokens = self._entry(role, rid, span_row)
            per_role[role] = span_gains(entry, tokens) if self.gain_unit == "span" else entry["step_gains"]
        if self.gain_source == "diff":
            scores = [t - s for t, s in zip(per_role["teacher"], per_role["student"])]
        else:
            scores = per_role[self.gain_source]
        if self.gain_unit == "span":
            n = len(scores)
            if n != len(span_row["step_of_span"]):
                raise ValueError(f"id {rid}: {n} span gains for {len(span_row['step_of_span'])} spans")
            return scores, [1] * n, list(range(n))
        return self._step_unit(span_row, scores)

    def _unit_weights(self, source, rid, bundle, tau, clip, lam, normalization, seed):
        scores, masses, unit_steps, restates = bundle
        kwargs = dict(masses=masses, tau=tau, clip=clip, lam=lam, normalization=normalization)
        if source == "alerd":
            if restates is not None:  # steps that restate the answer stay at weight 1
                per_unit = [restates[k] for k in self.spans[rid]["step_of_span"]] if self.gain_unit == "span" else restates
                return step_weights_for_source("answer-only-unit", scores, restates_answer=per_unit, **kwargs)
            return step_weights_for_source("gain", scores, **kwargs)
        if source == "random":
            return step_weights_for_source("random", scores,
                                           seed=random.Random(f"{seed}-{rid}").randrange(2 ** 31), **kwargs)
        return step_weights_for_source(source, [0.0] * len(masses), mean_logprobs=scores, **kwargs)

    @staticmethod
    def _restates_answer(row):
        """Steps whose text contains the gold answer (the last boxed answer of the solution)."""
        answer = last_boxed(row["output"])
        ends = row["step_char_ends"]
        starts = [0] + ends[:-1]
        return [bool(answer) and contains_answer(row["output"][a:b], answer) for a, b in zip(starts, ends)]

    # -- batches -----------------------------------------------------------------------------------------------

    def check(self, example_id, text_hash, teacher_hash, n_spans):
        row = self.spans.get(example_id)
        if row is None:
            raise ValueError(f"id {example_id} is not in the span index")
        if row["hash"] != text_hash or row["teacher_hash"] != teacher_hash:
            raise ValueError(f"id {example_id}: text differs from the one indexed (hash mismatch); rebuild spans.jsonl")
        if len(row["step_of_span"]) != n_spans:
            raise ValueError(f"id {example_id}: {n_spans} spans at train time, {len(row['step_of_span'])} indexed")

    def batch_weights(self, example_ids, hashes, teacher_hashes, n_spans, width):
        """(B, width) tensor of omega, zero-padded; stops on any hash or span-count mismatch."""
        out = torch.zeros(len(example_ids), width)
        for b, (rid, text_hash, teacher_hash, n) in enumerate(zip(example_ids, hashes, teacher_hashes, n_spans)):
            self.check(rid, text_hash, teacher_hash, n)
            out[b, :n] = torch.tensor(self.omega[rid])
        return out
