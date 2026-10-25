"""Shared fixtures: tiny offline tokenizers and synthetic GSM8K-like data."""
import random

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

WORDS = ("let triangle area hexagon find value sum answer therefore thus given right side length equals "
         "square root frac boxed we need to compute the number of ways 12 15 20 25 240 120 x y z").split()


def make_text(rng, sentences):
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12))).capitalize() + rng.choice([".", ".", "?"])
        for _ in range(sentences))


def train_tokenizer(kind, corpus, vocab_size=330, pad=True):
    """kind: bpe (byte-level, GPT-2 style) | wordpiece. pad=False leaves the tokenizer without a pad token, like GPT-2."""
    if kind == "bpe":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<eos>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    else:
        tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
        tok.normalizer = normalizers.Lowercase()
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordPieceTrainer(vocab_size=120, special_tokens=["<pad>", "<unk>"])
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>" if pad else None, eos_token="<eos>",
                                   unk_token="<unk>" if kind != "bpe" else None)
    fast.padding_side = "right"
    return fast


CHAT_TEMPLATE = ("{% for m in messages %}<|user|>\n{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def make_rows(rng, n, teacher_template=True):
    """SRA rows {id, question, answer, prompt, teacher_prompt, output, step_char_ends}: one line per step.

    The two prompts have different lengths (the teacher's carries a template header), as with the real models.
    """
    rows = []
    for i in range(n):
        question = make_text(rng, 2)
        lines = [make_text(rng, 1) for _ in range(rng.randint(2, 6))]
        answer = str(rng.randint(2, 999))
        lines.append(f"The final answer is \\boxed{{{answer}}}.")
        output, ends = "", []
        for line in lines:
            output += line + "\n"
            ends.append(len(output))
        output = output[:-1]
        ends[-1] = len(output)
        prompt = f"Q: {question}\nA:\n"
        teacher_prompt = f"<|user|>\nPlease think. {question}\n<|assistant|>\n" if teacher_template else prompt
        rows.append({"id": i, "question": question, "answer": answer, "prompt": prompt,
                     "teacher_prompt": teacher_prompt, "output": output, "step_char_ends": ends})
    return rows


def make_gain_file(rows, tokenizer, side, rng, model="tiny"):
    """A packed gain-tokens file with random token gains for `rows`, as token_gain.py would write it."""
    from token_gain import pack_gains, token_record

    records = [token_record(tokenizer, row, side) for row in rows]
    raw = {str(r["id"]): [rng.uniform(-1, 2) for _ in r["steps"]] for r in records}
    return pack_gains(records, raw, model, model)
