import importlib.util

import pytest

import verl.utils.attention_utils as attention_utils
import verl.utils.device as device_utils


def _reset_attention_utils_cache():
    attention_utils._index_first_axis = None
    attention_utils._pad_input = None
    attention_utils._rearrange = None
    attention_utils._unpad_input = None


def test_remove_padding_backend_unavailable_without_flash_attn(monkeypatch):
    _reset_attention_utils_cache()
    original_find_spec = importlib.util.find_spec

    monkeypatch.setattr(device_utils, "is_torch_npu_available", lambda check_device=False: False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "flash_attn" else original_find_spec(name),
    )

    assert attention_utils.is_remove_padding_backend_available() is False
    with pytest.raises(RuntimeError, match="`use_remove_padding=True` requires `flash_attn` on CUDA"):
        attention_utils.unpad_input(None, None)


def test_remove_padding_backend_available_on_npu_without_flash_attn(monkeypatch):
    _reset_attention_utils_cache()
    original_find_spec = importlib.util.find_spec

    monkeypatch.setattr(device_utils, "is_torch_npu_available", lambda check_device=False: True)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "flash_attn" else original_find_spec(name),
    )

    assert attention_utils.is_remove_padding_backend_available() is True
