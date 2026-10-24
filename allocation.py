"""Budget-preserving step weights (ALERD Eq. 3-5) and their projection onto SRA spans.

A step k holding l_k spans gets a weight w_k with sum_k l_k * w_k = N (N = number of spans), so
a span inherits omega_n = w_k(n) / N and sum_n omega_n = 1, the normalisation of SRA Eq. 8.
Ported from sgl.allocation.iwc.stable_iwc_step_weights; the mass l_k is a span count here.
"""
import math
import random
import sys

NORMALIZATIONS = ("per-trace", "corpus", "none")
SCORE_SOURCES = ("gain", "random", "predictability-easy", "predictability-hard", "answer-only-unit")


def last_boxed(text):
    """Content of the last \\boxed{...}, brace-matched (same rule as sgl.data.s1k.last_boxed)."""
    index = text.rfind("\\boxed{")
    if index == -1:
        return None
    start = index + len("\\boxed{")
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position].strip()
    return None


def exp_scores(scores, tau=1.0, clip=2.0, eps=1e-8):
    """Eq. 3: r_k = exp(clip(standardise(score), -c, c) / tau); uniform when the trace has no spread."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if clip < 0:
        raise ValueError("clip must be non-negative")
    if not scores:
        return []
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    if std < eps:
        return [1.0] * len(scores)
    scaled = [max(-clip, min(clip, (s - mean) / (std + eps))) / tau for s in scores]
    top = max(scaled)  # common shift keeps Eq. 4's ratios and stays finite for tiny tau
    return [max(math.exp(v - top), sys.float_info.min) for v in scaled]


def _check_masses(masses):
    """A step may hold no span (zero mass); the trace must still hold at least one."""
    if any(m < 0 for m in masses) or sum(masses) <= 0:
        raise ValueError("step span counts must be non-negative with a positive total")


def trace_weights(scores, masses, tau=1.0, lam=0.5, clip=2.0, eps=1e-8, normalization="per-trace",
                  corpus_ratio=None):
    """Eq. 3-5 for one trace. `corpus_ratio` = sum(l) / sum(l * r) over the corpus (normalization="corpus")."""
    if len(scores) != len(masses):
        raise ValueError("scores and masses must have the same number of steps")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    _check_masses(masses)
    r = exp_scores(scores, tau, clip, eps)
    if not r:
        return []
    if normalization == "per-trace":
        ratio = sum(masses) / sum(m * x for m, x in zip(masses, r))
    elif normalization == "corpus":
        if corpus_ratio is None:
            raise ValueError("corpus normalization needs corpus_ratio (see corpus_weights)")
        ratio = corpus_ratio
    else:
        ratio = 1.0
    return [(1.0 - lam) + lam * x * ratio for x in r]


def corpus_weights(traces, tau=1.0, lam=0.5, clip=2.0, eps=1e-8, normalization="per-trace"):
    """Weights for every (scores, masses) trace of a corpus; the corpus variant shares one rescaling factor."""
    ratio = None
    if normalization == "corpus":
        total = sum(sum(m) for _, m in traces)
        weighted = sum(sum(mk * x for mk, x in zip(m, exp_scores(s, tau, clip, eps))) for s, m in traces)
        ratio = total / weighted
    return [trace_weights(s, m, tau, lam, clip, eps, normalization, ratio) for s, m in traces]


def span_weights(step_w, step_of_span):
    """omega_n = w_{k(n)} / N for the spans of one sample (step_of_span indexes step_w)."""
    n = len(step_of_span)
    if n == 0:
        return []
    return [step_w[k] / n for k in step_of_span]


def step_scores(source, gains, *, mean_logprobs=None, restates_answer=None, seed=0):
    """Per-step scores for one trace under an ablation source; the allocation itself is unchanged.

    gain / answer-only-unit start from the answer gain; random permutes the gains within the trace;
    predictability-easy / -hard rank steps by the frozen student's mean token log-probability.
    Returns (scores, frozen) where frozen[k] marks steps held at unit weight (answer-only-unit).
    """
    if source not in SCORE_SOURCES:
        raise ValueError(f"source must be one of {SCORE_SOURCES}")
    frozen = [False] * len(gains)
    if source == "gain":
        return list(gains), frozen
    if source == "random":
        shuffled = list(gains)
        random.Random(seed).shuffle(shuffled)
        return shuffled, frozen
    if source == "answer-only-unit":
        if restates_answer is None or len(restates_answer) != len(gains):
            raise ValueError("answer-only-unit needs restates_answer per step")
        return list(gains), list(restates_answer)
    if mean_logprobs is None or len(mean_logprobs) != len(gains):
        raise ValueError("predictability sources need mean_logprobs per step")
    sign = 1.0 if source == "predictability-easy" else -1.0
    return [sign * v for v in mean_logprobs], frozen


def step_weights_for_source(source, gains, masses, *, mean_logprobs=None, restates_answer=None, seed=0,
                            tau=1.0, lam=0.5, clip=2.0, eps=1e-8, normalization="per-trace"):
    """Step weights for one trace; frozen steps keep weight 1 and the rest share the remaining budget."""
    scores, frozen = step_scores(source, gains, mean_logprobs=mean_logprobs,
                                 restates_answer=restates_answer, seed=seed)
    free = [k for k, f in enumerate(frozen) if not f]
    weights = [1.0] * len(gains)
    if free:
        sub = trace_weights([scores[k] for k in free], [masses[k] for k in free], tau, lam, clip, eps,
                            normalization if normalization != "corpus" else "per-trace")
        for k, w in zip(free, sub):
            weights[k] = w
    return weights
