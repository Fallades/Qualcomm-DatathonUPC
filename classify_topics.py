#!/usr/bin/env python3
"""Add first-pass mathematical topic labels to a JSONL M3Kang file.

Example:
  python3 classify_topics.py --input data/train.jsonl --output data/train_topics.jsonl
  python3 classify_topics.py --input data/val.jsonl --output data/val_topics.jsonl
  python3 classify_topics.py --input data/train.jsonl --output data/train_topics.jsonl \\
      --overrides data/topic_overrides.json --review data/topics_to_review.csv

Overrides are a JSON object: {"problem-id|language": "geometry", "problem-id": "algebra"}.
"""
import argparse
import csv
import json
import os
from collections import Counter

from copilot.topics import TOPICS, classify_problem


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overrides", default="", help="optional JSON file with manually checked labels")
    parser.add_argument("--review", default="", help="optional CSV listing low-confidence labels for manual review")
    args = parser.parse_args()

    overrides = {}
    if args.overrides:
        with open(args.overrides, encoding="utf-8") as stream:
            overrides = json.load(stream)
        if not isinstance(overrides, dict):
            parser.error("--overrides must contain a JSON object")

    problems = read_jsonl(args.input)
    tagged = [classify_problem(problem, overrides) for problem in problems]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        for problem in tagged:
            stream.write(json.dumps(problem, ensure_ascii=False) + "\n")

    counts = Counter(p["topic"] for p in tagged)
    confidence = Counter(p["topic_confidence"] for p in tagged)
    print(f"Tagged {len(tagged)} problems -> {args.output}")
    print("Topics:")
    for topic in TOPICS:
        print(f"  {topic:16} {counts[topic]:5}  {TOPICS[topic]}")
    print("Confidence: " + ", ".join(f"{k}={v}" for k, v in sorted(confidence.items())))

    if args.review:
        os.makedirs(os.path.dirname(os.path.abspath(args.review)), exist_ok=True)
        with open(args.review, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "language", "level", "topic", "confidence", "text"])
            for p in tagged:
                if p["topic_confidence"] == "low":
                    writer.writerow([p.get("id", ""), p.get("language", ""), p.get("level", ""),
                                     p["topic"], p["topic_confidence"], p.get("text", "")])
        print(f"Low-confidence labels for review -> {args.review}")


if __name__ == "__main__":
    main()
