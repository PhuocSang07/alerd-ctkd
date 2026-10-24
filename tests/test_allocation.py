import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from allocation import (corpus_weights, span_weights, step_weights_for_source,  # noqa: E402
                        trace_weights)

TOY_GAINS = [-0.69, 0.00, 2.30]


def test_alerd_toy_table():
    w = trace_weights(TOY_GAINS, [1, 1, 1], tau=1.0, lam=0.5, clip=2.0)
    assert [round(x, 2) for x in w] == [0.61, 0.70, 1.69]


def test_span_weights_sum_to_one():
    masses = [3, 1, 4]
    w = trace_weights(TOY_GAINS, masses)
    step_of_span = [0] * 3 + [1] * 1 + [2] * 4
    omega = span_weights(w, step_of_span)
    assert math.isclose(sum(omega), 1.0)
    assert math.isclose(sum(m * x for m, x in zip(masses, w)), sum(masses))


def test_lambda_zero_is_uniform():
    w = trace_weights(TOY_GAINS, [2, 5, 1], lam=0.0)
    assert w == [1.0, 1.0, 1.0]
    omega = span_weights(w, [0, 0, 1, 1, 1, 1, 1, 2])
    assert all(math.isclose(x, 1 / 8) for x in omega)


def test_flat_trace_is_uniform():
    assert trace_weights([0.3, 0.3, 0.3], [1, 2, 3]) == [1.0, 1.0, 1.0]


def test_corpus_normalization_preserves_corpus_mass_only():
    traces = [(TOY_GAINS, [1, 1, 1]), ([0.0, 0.1, 5.0, -1.0], [4, 1, 1, 2])]
    corpus = corpus_weights(traces, normalization="corpus")
    total = sum(sum(m) for _, m in traces)
    assert math.isclose(sum(mk * wk for (_, m), w in zip(traces, corpus) for mk, wk in zip(m, w)), total)
    per_trace = corpus_weights(traces, normalization="per-trace")
    for (_, m), w in zip(traces, per_trace):
        assert math.isclose(sum(mk * wk for mk, wk in zip(m, w)), sum(m))


def test_no_normalization_is_raw_weights():
    raw = trace_weights(TOY_GAINS, [1, 1, 1], normalization="none")
    per_trace = trace_weights(TOY_GAINS, [1, 1, 1])
    assert raw != per_trace and all(x > 0 for x in raw)


def test_random_source_permutes_gains():
    kw = dict(masses=[1, 1, 1, 1], seed=3)
    a = step_weights_for_source("random", [0.0, 1.0, 2.0, 3.0], **kw)
    b = step_weights_for_source("gain", [0.0, 1.0, 2.0, 3.0], **kw)
    assert sorted(round(x, 9) for x in a) == sorted(round(x, 9) for x in b)


def test_predictability_directions_are_opposite():
    lp = [-0.1, -1.0, -3.0]
    easy = step_weights_for_source("predictability-easy", [0, 0, 0], [1, 1, 1], mean_logprobs=lp)
    hard = step_weights_for_source("predictability-hard", [0, 0, 0], [1, 1, 1], mean_logprobs=lp)
    assert easy[0] > easy[2] and hard[2] > hard[0]


def test_answer_only_unit_freezes_restating_steps():
    w = step_weights_for_source("answer-only-unit", TOY_GAINS, [1, 1, 1], restates_answer=[False, False, True])
    assert w[2] == 1.0 and math.isclose(w[0] + w[1] + 1.0, 3.0)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        trace_weights([0.1], [0])
    with pytest.raises(ValueError):
        trace_weights([0.1], [1], lam=1.5)
