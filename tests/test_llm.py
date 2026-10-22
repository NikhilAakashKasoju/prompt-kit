from types import SimpleNamespace

import pytest

from src.prompt_kit import llm
from src.prompt_kit.llm import LLMOutputError, LLMUnavailableError, build_classification_model, generate, MAX_ATTEMPTS

def fake_reply(text):
    return SimpleNamespace(message=SimpleNamespace(content = text))

def text_generate_retires_after_invalid_reply(monkeypatch):
    replies = [
        '{"label": "shipping", "reasoning": "late"}',
        '{"label": "shipping", "confidence": 0.9, "reasoning": "late"}'
    ]
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return fake_reply(replies[len(calls) -1])

    monkeypatch.setattr(llm.ollama, "chat", fake_chat)

    model_cls = build_classification_model(["billing", "shipping", "unknown"])
    result = generate("some prompt", model_cls)

    assert result.label == "shipping"
    assert len(calls) == 2


def test_generate_raises_after_max_attempts(monkeypatch):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return fake_reply('{"label": "nonsense"}')      # a reply that is always invalid

    monkeypatch.setattr(llm.ollama, "chat", fake_chat)

    model_cls = build_classification_model(["billing", "shipping", "unknown"])

    with pytest.raises(LLMOutputError):
        generate("some prompt", model_cls)

    assert len(calls) == MAX_ATTEMPTS


def test_generate_retries_after_invalid_reply(monkeypatch):
    replies = [
        '{"label": "shipping", "reasoning": "late"}',  # invalid: no confidence
        '{"label": "shipping", "confidence": 0.9, "reasoning": "late"}',
    ]
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return fake_reply(replies[len(calls) - 1])

    monkeypatch.setattr(llm.ollama, "chat", fake_chat)

    model_cls = build_classification_model(["billing", "shipping", "unknown"])
    result = generate("some prompt", model_cls)

    assert result.label == "shipping"
    assert len(calls) == 2

def test_generate_raises_unavailable_when_server_down(monkeypatch):
    def fake_chat(**kwargs):
        raise ConnectionError("no server")

    monkeypatch.setattr(llm.ollama, "chat", fake_chat)

    model_cls = build_classification_model(["billing", "shipping", "unknown"])

    with pytest.raises(LLMUnavailableError):
        generate("some prompt", model_cls)