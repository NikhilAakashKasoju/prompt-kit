from fastapi import FastAPI, HTTPException

from .llm import LLMOutputError, classify
from .schemas import Classification, ClassifyRequest

app = FastAPI(title = "Prompt Kit")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model = Classification)
def classify_endpoint(req: ClassifyRequest):
    try:
        result = classify(req.text, req.labels)
    except LLMOutputError as e:
        raise HTTPException(status_code = 502, detail = str(e))

    return result.model_dump()

