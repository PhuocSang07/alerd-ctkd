r"""Word spans for the same-tokenizer case: SRA's matching of token end offsets, against word end offsets.

A word is a number with separators (`1,200`, `3.5`), a run of word characters, or one other non-space character; the
whitespace before a word belongs to it, as in the offsets of a BPE tokenizer. Lines are cut first and the whitespace after
the last word of a line (its line break) belongs to that word, so a word never reaches into the next line. A span
boundary is a point where a token boundary coincides with a word boundary (the longest_common_subsequence of data_utils);
a token that crosses several words merges them into one span. Every response token lies in exactly one span.

    word_span_starts(token_offsets, prompt, output, response_start) -> [response_start, e_1, ..., n_tokens]

Span n covers tokens [starts[n], starts[n + 1]). With one tokenizer for both models the teacher's spans are the
student's, shifted by the difference of the two prompt lengths in tokens (word_pair_bounds).
"""
import re

import numpy as np

from data_utils import sample_span_bounds

WORD = re.compile(r"\s*(?:\d+(?:[.,]\d+)*|\w+|[^\w\s])")
LINE = re.compile(r"[^\n]*\n|[^\n]+")


def word_end_offsets(text, char_start=0):
    """(n, 2) array of [start, end) character offsets of the words of `text`, shifted by char_start."""
    words = []
    for line in LINE.finditer(text):
        first = len(words)
        words.extend([m.start(), m.end()] for m in WORD.finditer(line.group(), 0))
        for word in words[first:]:
            word[0] += line.start()
            word[1] += line.start()
        if words:
            words[-1][1] = line.end()
    return np.array(words, dtype=np.int64).reshape(-1, 2) + char_start


def word_span_starts(token_offsets, prompt, output, response_start, max_spans=None):
    """Token index where each span starts, then the number of tokens: the last span reaches the last token."""
    n_tokens = len(token_offsets)
    words = word_end_offsets(output, len(prompt))
    bounds = sample_span_bounds(token_offsets, words, response_start, 0, max_spans)
    starts = [int(b[0]) for b in bounds]
    if starts[-1] < n_tokens:
        starts.append(n_tokens)
    return starts


def word_pair_bounds(token_offsets, teacher_length, response_start, teacher_start, prompt, output, max_spans=None):
    """(student_idx, teacher_idx) span boundaries like sample_span_bounds, for a student and a teacher that share a
    tokenizer. The teacher's response must tokenize to as many tokens as the student's."""
    n_tokens = len(token_offsets)
    if int(teacher_length) - int(teacher_start) != n_tokens - int(response_start):
        raise ValueError("word spans need one tokenizer for both models: the response has "
                         f"{n_tokens - int(response_start)} student tokens and {int(teacher_length) - int(teacher_start)} "
                         "teacher tokens")
    shift = int(teacher_start) - int(response_start)
    return [(s, s + shift) for s in word_span_starts(token_offsets, prompt, output, int(response_start), max_spans)]
