import json 
import time
from pathlib import Path

from src.prompt_kit.llm import LLMOutputError, classify, extract

CASES_FILE = Path(__file__).parent / "cases.json"

def eval_classify(section):
    labels = section["labels"]
    correct = 0
    failures = 0
    latencies = []

    for case in section["cases"]:
        start = time.perf_counter()
        try:
            result = classify(case["text"], labels)
        except LLMOutputError:
            failures += 1
            print(f"    Error   {case['text']!r}")
            continue
        latencies.append(time.perf_counter() - start)

        ok = result.label == case["expected"]
        correct += ok
        mark = "ok  " if ok else "MISS"
        print(f"    {mark}  {case['text']!r} -> {result.label} ({result.confidence})")

    total = len(section["cases"])
    print(f"\nclassify: {correct}/{total} correct, {failures} invalid-outpout failures")
    if latencies:
        print(f"avg_latency: {sum(latencies)/ len(latencies):.1f}s")

def normalize(value):
    return None if value is None else str(value).strip().lower()

def eval_extract(section):
    print("cases:", len(section["cases"]))
    total = 0
    correct = 0
    null_fields = 0
    hallucinated = 0
    failures = 0

    for case in section["cases"]:
        try:
            result = extract(case["text"], case["fields"])
        except LLMOutputError:
            failures += 1
            print(f"    Error   {case['text']!r}")
            continue

        for field in case["fields"]:
            want = normalize(case["expected"][field])
            got = normalize(result.data.get(field))
            total += 1

            ok = got == want
            correct += ok

            if want is None:
                null_fields += 1
                if got is not None:
                    hallucinated += 1

            if not ok:
                print(f"    MISS {field}: expected {want!r}, got {got!r}")

    print(f"\nextract: {correct}/{total} fields correct, {failures} invalid-output failures")
    print(f"hallucinated: {hallucinated}/{null_fields} null fields were filled in")


if __name__ == "__main__":
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print("== classify ==")
    eval_classify(data["classify"])
    print("== extract ==")
    eval_extract(data["extract"])
