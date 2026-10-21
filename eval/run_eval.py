import json 
import time
from pathlib import Path

from src.prompt_kit.llm import LLMOutputError, classify

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


if __name__ == "__main__":
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print("== classify ==")
    eval_classify(data["classify"])

