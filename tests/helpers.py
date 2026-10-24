"""Shared fixtures: tiny offline tokenizers and synthetic math-like text."""
import random

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

WORDS = ("let triangle area hexagon find value sum answer therefore thus given right side length equals "
         "square root frac boxed we need to compute the number of ways 12 15 20 25 240 120 x y z").split()


def make_text(rng, sentences):
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12))).capitalize() + rng.choice([".", ".", "?"])
        for _ in range(sentences))


def train_tokenizer(kind, corpus, vocab_size=330):
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
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>" if kind != "bpe" else None)
    fast.padding_side = "right"
    return fast




def make_rows(rng, n):
    """SRA rows {id, prompt, output, step_char_ends}: one sentence per step."""
    rows = []
    for i in range(n):
        prompt = make_text(rng, 2) + "\n"
        sentences = [make_text(rng, 1) for _ in range(rng.randint(2, 6))]
        output, ends = "", []
        for sentence in sentences:
            output += sentence + " "
            ends.append(len(output))
        rows.append({"id": i, "prompt": prompt, "output": output, "step_char_ends": ends})
    return rows
