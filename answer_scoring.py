"""Final-answer extraction and Math-Verify grading shared by answer-gain scoring and evaluation."""
import os
import re
import signal
from contextlib import contextmanager

VERIFY_TIMEOUT_SECONDS = 10
_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed(text):
    """Content of the last \\boxed{...}, with brace matching for nested expressions."""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, index = 1, start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[start:index - 1].strip() if depth == 0 else text[start:].strip()


@contextmanager
def _time_limit(seconds):
    """Abort a hanging symbolic verification. POSIX only; elsewhere the call runs without a limit."""
    if os.name != "posix":
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("verification timed out")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def grade_math_verify(predictions, gold):
    """0/1 per generation: Math-Verify on the generation against `$gold$`. A timeout or parse error counts as wrong."""
    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise ImportError("grading needs the math-verify package (pip install math-verify)") from error
    try:
        with _time_limit(VERIFY_TIMEOUT_SECONDS):
            parsed_gold = parse("$" + gold + "$")
            return [int(bool(verify(parsed_gold, parse(prediction)))) for prediction in predictions]
    except Exception:
        return [0] * len(predictions)
