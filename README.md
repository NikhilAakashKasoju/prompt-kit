# Prompt Kit

A FastAPI service backed by a local Ollama model that summarizes text, extracts structured fields, and classifies text, returning validated JSON.

> **Status:** work in progress. The three endpoints work and have evals and tests. Hardening (error handling, logging, config) and Docker are next.

## Why this exists

LLMs return free text, but software needs predictable data. Prompt Kit wraps a local Ollama model with schema-constrained output, validation and retries, so any app can call it over HTTP and rely on getting valid JSON back. Everything runs locally: no API keys, no per-call cost, and no data leaves your machine.

## Goals

- Run fully local with a small model (`llama3.2:3b` via Ollama)
- Every response is schema-valid JSON, enforced with Pydantic
- Retry on invalid model output, and return a clean error if it still fails
- Measure quality with a small eval set

## Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | none | `{"status": "ok"}` |
| `POST /summarize` | `text`, `max_sentences` (1 to 10, default 3) | `summary`, `key_points` |
| `POST /extract` | `text`, `fields` (list of names) | `data`: a value (or `null`) per requested field |
| `POST /classify` | `text`, `labels` (at least 2) | `label`, `confidence`, `reasoning` |

How it works:

- `/classify` builds its schema from the caller's labels and adds an `unknown` label, so the model can only return one of those values.
- `/extract` builds its schema from the requested field names, so the model must return every requested key, using `null` when the text does not state it.
- Replies that fail validation are retried (up to 3 attempts) with the validation error fed back to the model.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com/).

```bash
ollama pull llama3.2:3b
uv sync
```

Make sure Ollama is running before starting the service.

## Run the service

```bash
uv run uvicorn src.prompt_kit.main:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive documentation where you can try each endpoint.

Example request body for `POST /classify`:

```json
{"text": "I was charged twice", "labels": ["billing", "shipping"]}
```

## Run tests

```bash
uv run pytest -v
```

The tests do not need Ollama. The model is replaced with a fake, so they run in about a second.

## Run the eval

```bash
uv run python -m eval.run_eval
```

This calls the real model on the cases in `eval/cases.json` and prints accuracy, hallucination counts and latency. It needs Ollama running.

## Project structure

```
src/prompt_kit/
  schemas.py   # Pydantic request and response models
  llm.py       # Ollama calls, dynamic schemas, validation and retry
  main.py      # FastAPI app and endpoints
eval/
  cases.json   # test inputs and expected outputs
  run_eval.py  # scores classify, extract and summarize
tests/         # pytest tests (schemas, retry logic, API)
```

## Eval results

| Task | Cases | Result | Avg latency |
|---|---|---|---|
| classify | 16 | 15/16 correct | 2.3s |
| extract | 5 (19 fields) | 17/19 fields correct, 0/5 hallucinated | n/a |
| summarize | 3 | 3/3 within sentence limit, 5/6 required words kept | 3.5s |

These sets are small, and the prompts were tuned while looking at them, so treat the numbers as a baseline and not as a promise of real-world accuracy.

## Known limitations

- The model's confidence scores are not calibrated: wrong answers often score the same as right ones.
- Texts that mix two topics (for example, a charge for shipping) can land on either label.
- The model sometimes returns `null` for a value that is in the text (missed, not invented).
- A 3B model suits low-stakes sorting and drafting. Use a larger model, or human review, where a wrong answer is costly.

## Progress

- [x] Project setup with uv
- [x] Request and response schemas
- [x] Schema tests
- [x] First Ollama call with schema-constrained JSON output
- [x] Restrict classification labels to the caller's list
- [x] Validation and retry logic
- [x] FastAPI endpoints (`/classify`, `/summarize`, `/extract`)
- [x] Eval set and scoring script
- [x] API tests with a mocked model
- [ ] Clear 503 when the model server is down
- [ ] Prompts in separate files
- [ ] Logging
- [ ] Config through environment variables
- [ ] Dockerfile

## Notes

- Ollama's default context window can exhaust RAM on modest machines, so calls set `num_ctx` explicitly (4096).
- Ollama silently ignores misspelled option names, so options are kept in one place.
- Response fields with defaults can be skipped by the model, so required fields have no default.
- Telling the model to "copy values exactly as written" fixed a bug where multi-field extraction garbled a currency value.
- Narrow prompt rules beat blanket ones: a rule that sent every payment mention to `billing` fixed one case and broke two.