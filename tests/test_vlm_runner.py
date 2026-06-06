from __future__ import annotations

import builtins

import pytest

from src.vlm.dummy import DummyVLMRunner
from src.vlm.qwen2_5_vl import Qwen25VLRunner


def test_dummy_vlm_runner_returns_empty() -> None:
    runner = DummyVLMRunner()
    assert runner.generate(None, "Question?") == ""
    assert runner.skipped is True


def test_qwen_runner_dependency_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "qwen_vl_utils":
            raise ImportError("missing qwen_vl_utils")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Qwen2.5-VL dependencies are not installed"):
        Qwen25VLRunner()
