import json 
import time
from pathlib import Path
import re
from src.prompt_kit.llm import LLMOutputError, classify, extract, summarize

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
    print(f"\nclassify: {correct}/{total} correct, {failures} invalid-output failures")
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

def count_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if p])


def eval_summarize(section):
    within_limit = 0
    has_points = 0
    mentions_found = 0
    mentions_total = 0
    failures = 0
    latencies = []

    for case in section["cases"]:
        start = time.perf_counter()
        try:
            result = summarize(case["text"], case["max_sentences"])
        except LLMOutputError:
            failures += 1
            print(f"    Error   {case['text'][:50]!r}")
            continue
        latencies.append(time.perf_counter() - start)

        n = count_sentences(result.summary)
        if n <= case["max_sentences"]:
            within_limit += 1
        else:
            print(f"    MISS  {n} sentences, limit {case['max_sentences']}: {result.summary!r}")

        if len(result.key_points) > 0:
            has_points += 1
        else:
            print(f"    MISS  no key points for {case['text'][:50]!r}")

        for word in case["must_mention"]:
            mentions_total += 1
            if word.lower() in result.summary.lower():
                mentions_found += 1
            else:
                print(f"    MISS  summary does not mention {word!r}")

    total = len(section["cases"])
    print(f"\nsummarize: {within_limit}/{total} within sentence limit, {has_points}/{total} have key points")
    print(f"coverage: {mentions_found}/{mentions_total} required words kept, {failures} invalid-output failures")
    if latencies:
        print(f"avg_latency: {sum(latencies) / len(latencies):.1f}s")

if __name__ == "__main__":
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print("== classify ==")
    eval_classify(data["classify"])
    print("== extract ==")
    eval_extract(data["extract"])
    print("== summarize ==")
    eval_summarize(data["summarize"])