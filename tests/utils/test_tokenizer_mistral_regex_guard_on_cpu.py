from pathlib import Path

from verl.utils.tokenizer import hf_tokenizer


def test_qwen3_explicitly_opts_out_of_mistral_regex_fix(monkeypatch, tmp_path: Path):
    calls = []

    class Config:
        model_type = "qwen3"

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Config()

    class Tokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            return Tokenizer()

    monkeypatch.setattr("transformers.AutoConfig", AutoConfig)
    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)

    hf_tokenizer(str(tmp_path), trust_remote_code=True)

    assert calls == [{"trust_remote_code": True, "fix_mistral_regex": False}]


def test_mistral_does_not_override_transformers_regex_decision(monkeypatch, tmp_path: Path):
    calls = []

    class Config:
        model_type = "mistral"

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Config()

    class Tokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            return Tokenizer()

    monkeypatch.setattr("transformers.AutoConfig", AutoConfig)
    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)

    hf_tokenizer(str(tmp_path), trust_remote_code=True)

    assert calls == [{"trust_remote_code": True}]
