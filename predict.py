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
import base64
import csv
import json
import os
import sys

from copilot.config import CONFIG
from copilot.pipeline import solve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rag", type=int, default=0, metavar="K", help="add K similar solved problems")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--vision", action="store_true",
                   help="send available problem images to the local multimodal server")
    p.add_argument("--images", default="data/images",
                   help="folder with <id>_<language>.png files (used only with --vision)")
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
            kwargs = {}
            if args.vision:
                image_name = f"{problem['id']}_{problem.get('language', 'eng')}.png"
                image_path = os.path.join(args.images, image_name)
                if os.path.isfile(image_path):
                    with open(image_path, "rb") as image_file:
                        kwargs = {
                            "image_b64": base64.b64encode(image_file.read()).decode("ascii"),
                            "llm_url": f"http://127.0.0.1:{CONFIG['VISION_PORT']}/v1/chat/completions",
                        }
            result = solve(problem, index=index, k=args.rag, **kwargs)
            writer.writerow([problem["id"], problem.get("language", ""), result["answer"],
                             result["seconds"], result["status"]])
            out.flush()
            print(f"[{n}/{len(problems)}] {problem['id']} -> {result['answer']} "
                  f"in {result['seconds']:.1f} s ({result['status']})", file=sys.stderr)


if __name__ == "__main__":
    main()
