from fastapi.testclient import TestClient

from src.prompt_kit import main
from src.prompt_kit.llm import LLMOutputError, LLMUnavailableError
from src.prompt_kit.schemas import Classification, Extraction

client = TestClient(main.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_returns_result(monkeypatch):
    def fake_classify(text, labels):
        return Classification(label="shipping", confidence=0.9, reasoning="late order")

    monkeypatch.setattr(main, "classify", fake_classify)

    resp = client.post(
        "/classify",
        json={"text": "My order is late", "labels": ["billing", "shipping"]},
    )

    assert resp.status_code == 200
    assert resp.json()["label"] == "shipping"


def test_classify_rejects_bad_input():
    resp = client.post("/classify", json={"text": "", "labels": ["billing"]})
    assert resp.status_code == 422


def test_classify_returns_502_when_model_fails(monkeypatch):
    def failing_classify(text, labels):
        raise LLMOutputError("model failed")

    monkeypatch.setattr(main, "classify", failing_classify)

    resp = client.post(
        "/classify",
        json={"text": "My order is late", "labels": ["billing", "shipping"]},
    )

    assert resp.status_code == 502

def test_summarize_rejects_zero_sentences():
    resp = client.post("/summarize", json={"text": "hello", "max_sentences": 0})
    assert resp.status_code == 422


def test_extract_returns_data(monkeypatch):
    def fake_extract(text, fields):
        return Extraction(data={"name": "Priya", "email": None})

    monkeypatch.setattr(main, "extract", fake_extract)

    resp = client.post(
        "/extract",
        json={"text": "Priya works here.", "fields": ["name", "email"]},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["email"] is None

def test_classify_returns_503_when_server_down(monkeypatch):
    def unavailable_classify(text, labels):
        raise LLMUnavailableError("server down")

    monkeypatch.setattr(main, "classify", unavailable_classify)

    resp = client.post(
        "/classify",
        json={"text": "My order is late", "labels": ["billing", "shipping"]},
    )

    assert resp.status_code == 503