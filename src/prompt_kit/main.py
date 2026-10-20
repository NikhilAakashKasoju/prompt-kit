from fastapi import FastAPI, HTTPException

from .llm import LLMOutputError, classify, summarize, extract
from .schemas import Classification, ClassifyRequest, Summary, SummarizeRequest, Extraction, ExtractRequest

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

@app.post("/summarize", response_model = Summary)
def summary_endpoint(req: SummarizeRequest):
    try:
        result = summarize(req.text, req.max_sentences)
    except LLMOutputError as e:
        raise HTTPException(status_code = 502, detail = str(e))

    return result.model_dump()

@app.post("/extract", response_model = Extraction)
def extract_endpoint(req: ExtractRequest):
    try:
        result = extract(req.text, req.fields)
    except LLMOutputError as e:
        raise HTTPException(status_code = 502, detail = str(e))

    return result.model_dump()

