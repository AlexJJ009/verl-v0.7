from pathlib import Path

import pandas as pd
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class FakeQwenThinkingTokenizer:
    pad_token_id = 0

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join(chr(token_id) for token_id in token_ids if token_id != 0)

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        rendered = ""
        for message in messages:
            content = message["content"]
            if message["role"] == "assistant" and len(messages) == 1 and "</think>" in content:
                content = content.split("</think>", 1)[1].lstrip()
            rendered += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"

        input_ids = self._encode(rendered)
        if not tokenize:
            return rendered
        if not return_dict:
            return input_ids

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.ones_like(input_ids_tensor),
        }


def test_whole_message_tokenization_supervises_qwen_thinking_content(tmp_path: Path):
    reasoning = "Derive the result step by step."
    messages = [
        {"role": "system", "content": "Use <think> and <answer>."},
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n<answer>\\boxed{{2}}</answer>",
        },
    ]
    data_file = tmp_path / "qwen_thinking.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(data_file)

    tokenizer = FakeQwenThinkingTokenizer()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(data_file),
        tokenizer=tokenizer,
        config={
            "messages_key": "messages",
            "max_length": 2048,
            "pad_mode": "no_padding",
            "truncation": "error",
            "tokenize_whole_message": True,
        },
    )

    item = dataset[0]
    supervised_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1])
    full_text = tokenizer.decode(item["input_ids"])

    assert full_text == tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assert "<think>" in supervised_text
    assert reasoning in supervised_text
    assert "</think>" in supervised_text
    assert "<answer>\\boxed{2}</answer>" in supervised_text
    assert "<|im_end|>" in supervised_text
    assert "Use <think> and <answer>." not in supervised_text
    assert "What is 1+1?" not in supervised_text


def test_whole_message_tokenization_handles_qwen_context_sensitive_multi_turn(tmp_path: Path):
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "<think>\nfirst reasoning\n</think>\n<answer>first</answer>"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "<think>\nsecond reasoning\n</think>\n<answer>second</answer>"},
    ]
    data_file = tmp_path / "qwen_thinking_multi_turn.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(data_file)

    tokenizer = FakeQwenThinkingTokenizer()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(data_file),
        tokenizer=tokenizer,
        config={
            "messages_key": "messages",
            "max_length": 2048,
            "pad_mode": "no_padding",
            "truncation": "error",
            "tokenize_whole_message": True,
        },
    )

    item = dataset[0]
    supervised_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1])

    assert "first reasoning" in supervised_text
    assert "<answer>first</answer>" in supervised_text
    assert "second reasoning" in supervised_text
    assert "<answer>second</answer>" in supervised_text
    assert supervised_text.count("<|im_end|>") == 2
    assert "First question" not in supervised_text
    assert "Second question" not in supervised_text
