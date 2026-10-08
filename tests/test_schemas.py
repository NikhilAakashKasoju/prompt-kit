import pytest
from pydantic import ValidationError
from src.prompt_kit.schemas import SummarizeRequest, Extraction, Classification

def test_summarize_default_sentences():
    req = SummarizeRequest(text = "hello")
    assert req.max_sentences == 3

def test_summarize_rejects_empty_text():
    with pytest.raises(ValidationError):
        SummarizeRequest(text = "")

def test_summarize_rejects_zero_sentences():
    with pytest.raises(ValidationError):
        SummarizeRequest(text = "hello", max_sentences = 0)

def test_extraction_input_valid_dictionary():
    req = Extraction.model_validate_json('{"data": {"name": "john"}}')
    assert req.data["name"] == "john"

def test_extraction_default_data():
    req = Extraction.model_validate_json('{"data": {"name": null, "age": "15"}}')
    assert req.data["name"] == None
    assert req.data["age"] == "15"

def test_extraction_rejects_data_as_list():
    #req = Extraction.model_validate_json('{"data": ["name": "john", "age": "15"]}')
    with pytest.raises(ValidationError):
        Extraction.model_validate_json('{"data": ["john","15"]}')

def test_classification_requires_confidence():
    with pytest.raises(ValidationError):
        Classification.model_validate_json(
            '{"lable": "shipping", "reasoning", "late order"}'
        )