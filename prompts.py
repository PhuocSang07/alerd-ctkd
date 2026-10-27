r"""Prompt text shared by data preparation, gain scoring, training and evaluation.

The instruction comes first with no separator before the question. A tokenizer with a chat
template wraps it as a single user turn; a tokenizer without one (GPT-2) gets the instruction as raw text, so every model
sees the same words and only the wrapping differs. Special tokens come from the template, so the prompt is tokenized
with add_special_tokens=False everywhere.
"""

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}.{question}"

# Llama 3.x templates print today's date unless told otherwise; a fixed date keeps train and eval prompts identical.
FIXED_DATE = "26 Jul 2024"


def has_chat_template(tokenizer):
    return bool(getattr(tokenizer, "chat_template", None))


def wrap_user(tokenizer, content):
    """One user turn in the tokenizer's chat template (raw text when it has none)."""
    if not has_chat_template(tokenizer):
        return content
    template = tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else ""
    extra = {"date_string": FIXED_DATE} if "date_string" in template else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False, **extra)


def build_prompt(tokenizer, question):
    """Raw instruction for tokenizers without a chat template, else the instruction as one templated user turn."""
    return wrap_user(tokenizer, INSTRUCTION.format(question=question))
