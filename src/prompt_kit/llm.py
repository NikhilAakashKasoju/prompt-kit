import ollama
from typing import Literal
from pydantic import Field, create_model, ValidationError
from .schemas import Summary, Extraction

MODEL = "llama3.2:3b"
OPTIONS = {"num_ctx": 4096, "temperature": 0}


def build_classification_model(labels: list[str]):
    return create_model(
        "DynamicClassification",
        label = (Literal[tuple(labels)], ...),
        confidence = (float, Field(ge = 0, le = 1)),
        reasoning = (str, Field(min_length = 1)),
    )

def classify(text: str, labels: list[str]):
    allowed = labels + ["unknown"]
    model_cls = build_classification_model(allowed)
    prompt = (
        f"Classify the text into exactly one of these labels: {', '.join(allowed)}.\n"
        f"Use 'unknown' for greetings, general questions, or anything that does not "
        f"clearly belong to one of the other labels.\n"
        f"Choose billing only when the main issue is a charge, payment or refund.\n\n"
        f"Text: {text}"
    )

    return generate(prompt, model_cls)

def summarize(text: str, max_sentences: int):
    prompt = (
        f"Summarize the text in at most {max_sentences} sentences.\n"
        f"Also list the key points as short bullet strings.\n\n"
        f"Text: {text}"
    )
    return generate(prompt, Summary)

def build_extraction_model(fields: list[str]):
    return create_model(
        "DynamicExtraction",
        **{name: (str | None, ...) for name in fields},
    )
NULL_STRINGS = {"null", "none", "n/a", "na", ""}

def clean_value(value):
    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return None
    return value

def extract(text: str, fields: list[str]):
    model_cls = build_extraction_model(fields)
    prompt = (
        f"Extract these fields from the text: {', '.join(fields)}.\n"
        f"Use only information stated in the text.\n"
        f"Copy each value exactly as written in the text.\n"
        f"If a field is not stated, use null. Do not guess.\n\n"
        f"Text: {text}"
    )
    result = generate(prompt, model_cls)
    cleaned = {k: clean_value(v) for k, v in result.model_dump().items()}
    return Extraction(data = cleaned)

MAX_ATTEMPTS = 3

class LLMOutputError(Exception):
    """Raised when the model never produces valid output."""
    
def generate(prompt: str, model_cls):
    messages = [{"role": "user", "content": prompt}]
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        resp = ollama.chat(
            model = MODEL,
            messages = messages,
            format = model_cls.model_json_schema(),
            options = OPTIONS,
        )

        raw = resp.message.content

        try:
            return model_cls.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"your reply was invalid: {e}\n Return corrected JSON only."
            })

    raise LLMOutputError(
        f"Model failed to produce valid output after {MAX_ATTEMPTS} attempts: {last_error}"
    )

