# Prompt Toolkit

A FastAPI service backed by a local Ollama model that summarizes text, extracts structured fields, and classifies text, returning validated JSON.

> **Status:** work in progress. Schemas and tests are done; the LLM layer and API endpoints are next.

## Goals

- Run fully local with a small model (`llama3.2:3b` via Ollama)
- Every response is schema-valid JSON, enforced with Pydantic
- Retry on invalid model output, and return a clean error if it still fails
- Measure quality with a small eval set

## Planned endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /summarize` | text, max sentences | summary and key points |
| `POST /extract` | text, list of fields | value (or null) per field |
| `POST /classify` | text, list of labels | label, confidence, reasoning |

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com/).

```bash
ollama pull llama3.2:3b
uv sync
```

## Run tests

```bash
uv run pytest -v
```

## Project structure

```
src/prompt_kit/
  schemas.py   # Pydantic request and response models
  llm.py       # Ollama calls, validation, retry (in progress)
  main.py      # FastAPI app (in progress)
  prompts/     # prompt templates
tests/         # pytest tests
```

## Progress

- [x] Project setup with uv
- [x] Request and response schemas
- [x] Schema tests
- [x] First Ollama call with schema-constrained JSON output
- [ ] Restrict classification labels to the caller's list
- [ ] Validation and retry logic
- [ ] FastAPI endpoints
- [ ] Eval set and scoring script
- [ ] Dockerfile

## Notes

- Ollama's default context window can exhaust RAM on modest machines, so calls set `num_ctx` explicitly (4096).
- Ollama silently ignores misspelled option names, so options are kept in one place.
- Response fields with defaults can be skipped by the model, so required fields must have no default.