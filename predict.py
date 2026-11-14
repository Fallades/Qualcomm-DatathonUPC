#!/usr/bin/env python3
"""Official entry point: answer every problem with one letter, A to E.

    python3 predict.py --input problems.jsonl --output predictions.csv
    python3 predict.py --input problems.jsonl --output predictions.csv --rag 1

Input: one JSON object per line with at least "id" and "text", plus
"options" ({"A": ..., "E": ...}) and "language" when available.
Output: a CSV with id, language, answer, seconds, status.

Rules: runs offline, at most 60 seconds per problem, deterministic.
"""
import argparse
import csv
import json
import os
import sys

from copilot.pipeline import solve
from copilot.config import CONFIG
from copilot.images import load_b64
from copilot.llm import URL

VISION_URL = f"http://127.0.0.1:{CONFIG['VISION_PORT']}/v1/chat/completions"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rag", type=int, default=0, metavar="K", help="add K similar solved problems")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--images", default="data/images",
                   help="problem images used for non-text-only problems")
    p.add_argument("--no-vision", action="store_true",
                   help="do not send non-text-only problem images")
    args = p.parse_args()

    index = None
    if args.rag:
        if not os.path.exists(args.index):
            sys.exit(f"{args.index} not found. Build it first: python3 build_index.py")
        from copilot.rag import Index
        index = Index(args.index)

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    with open(args.output, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "answer", "seconds", "status"])
        for n, problem in enumerate(problems, 1):
            image = None
            vision_url = None
            if not args.no_vision and not problem.get("text_only", True):
                image = load_b64(problem, args.images)
                if image:
                    vision_url = VISION_URL
                else:
                    print(f"warning: image not found for {problem['id']}", file=sys.stderr)
            result = solve(problem, index=index, k=args.rag, image_b64=image,
                           llm_url=vision_url or URL)
            writer.writerow([problem["id"], problem.get("language", ""), result["answer"],
                             result["seconds"], result["status"]])
            out.flush()
            print(f"[{n}/{len(problems)}] {problem['id']} -> {result['answer']} "
                  f"in {result['seconds']:.1f} s ({result['status']})", file=sys.stderr)


if __name__ == "__main__":
    main()
