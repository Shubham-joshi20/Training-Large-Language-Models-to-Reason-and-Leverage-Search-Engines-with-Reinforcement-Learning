"""Prompt formatting shared by training and inference."""

from __future__ import annotations


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Use the model's chat template when one is available."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
