from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .llm import LLMOutputError, LLMUnavailableError, classify, summarize, extract
from .schemas import Classification, ClassifyRequest, Summary, SummarizeRequest, Extraction, ExtractRequest

app = FastAPI(title = "Prompt Kit")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model = Classification)
def classify_endpoint(req: ClassifyRequest):
    return classify(req.text, req.labels).model_dump()

@app.post("/summarize", response_model = Summary)
def summary_endpoint(req: SummarizeRequest):
    return summarize(req.text, req.max_sentences).model_dump()

@app.post("/extract", response_model = Extraction)
def extract_endpoint(req: ExtractRequest):
    return extract(req.text, req.fields).model_dump()

@app.exception_handler(LLMOutputError)
def handle_output_error(request, exc):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(LLMUnavailableError)
def handle_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})