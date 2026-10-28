import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CATEGORIES = ["Billing", "Technical", "Account"]


def load_prompt(version):
    with open(f"prompts/classifier_{version}.txt", encoding="utf-8") as f:
        return f.read()


def load_golden(path="golden.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify(prompt, ticket):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Ticket: {ticket}",
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0,
        ),
    )
    text = (response.text or "").lower()
    found = [(text.find(c.lower()), c) for c in CATEGORIES if c.lower() in text]
    return min(found)[1] if found else "Unknown"


def run_eval(version):
    prompt = load_prompt(version)
    golden = load_golden()
    correct = 0
    for item in golden:
        predicted = classify(prompt, item["ticket"])
        if predicted == item["category"]:
            correct += 1
        else:
            print(f"  [{version}] miss: {item['ticket']} -> {predicted} (expected {item['category']})")
        time.sleep(1)  # avoid rate limits
    total = len(golden)
    pct = int(correct / total * 100)  # int() truncates: 8/12 -> 66, 11/12 -> 91
    print(f"{version}: {correct}/{total} ({pct}%)")
    return correct, total, pct


if __name__ == "__main__":
    versions = sys.argv[1:] or ["v1", "v2"]
    results = {v: run_eval(v) for v in versions}
    print(" | ".join(f"{v}: {c}/{t} ({p}%)" for v, (c, t, p) in results.items()))