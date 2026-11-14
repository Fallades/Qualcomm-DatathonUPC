#!/usr/bin/env python3
"""Measure accuracy and latency on the board, the way the official run does.

    python3 eval.py --limit 50                    # baseline on 50 validation problems
    python3 eval.py --limit 50 --rag 1 --name rag1
    python3 eval.py --languages cat --limit 100

Every run writes per-problem results to results/ and adds one line to
results/ablations.csv, so you can build your ablation table as you go.
"""
import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

from copilot.config import CONFIG
from copilot.pipeline import TIME_LIMIT, solve
from copilot.images import load_b64
from copilot.llm import URL

VISION_URL = f"http://127.0.0.1:{CONFIG['VISION_PORT']}/v1/chat/completions"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def table(title, groups):
    print(f"\n{title}")
    for key in sorted(groups, key=str):
        hits = groups[key]
        print(f"  {str(key):12} {sum(hits):4}/{len(hits):<4} {100 * sum(hits) / len(hits):5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl")
    p.add_argument("--limit", type=int, default=50, help="0 means every problem")
    p.add_argument("--languages", default="", help="e.g. eng,cat")
    p.add_argument("--rag", type=int, default=0, metavar="K")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--images", default="data/images",
                   help="problem images used for non-text-only problems")
    p.add_argument("--no-vision", action="store_true",
                   help="do not send non-text-only problem images")
    p.add_argument("--name", default="", help="a label for this experiment in ablations.csv")
    args = p.parse_args()

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    languages = set(filter(None, args.languages.split(",")))
    if languages:
        problems = [x for x in problems if x["language"] in languages]
    random.Random(42).shuffle(problems)  # the same sample every run, so runs compare
    if args.limit:
        problems = problems[:args.limit]

    index = None
    if args.rag:
        from copilot.rag import Index
        index = Index(args.index)

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or (f"rag{args.rag}" if args.rag else "baseline")
    detail_path = f"results/eval-{stamp}-{name}.csv"
    reasoning_path = f"results/vision-reasoning-{stamp}-{name}.jsonl"
    crop_dir = f"results/crops-{stamp}-{name}"

    rows = []
    with open(detail_path, "w", newline="") as out:
        with open(reasoning_path, "w", encoding="utf-8") as reasoning_out:
            writer = csv.writer(out)
            writer.writerow(["id", "language", "level", "has_options", "label", "answer",
                             "correct", "seconds", "status", "prompt_tokens", "retrieved"])
            for n, problem in enumerate(problems, 1):
                image = None
                vision_url = None
                if not args.no_vision and not problem.get("text_only", True):
                    image = load_b64(problem, args.images, save_dir=crop_dir)
                    if image:
                        vision_url = VISION_URL
                    else:
                        print(f"warning: image not found for {problem['id']}", file=sys.stderr)
                result = solve(problem, index=index, k=args.rag, exclude_id=problem["id"],
                               image_b64=image, llm_url=vision_url or URL)
                correct = result["answer"] == problem["label"]
                row = {**problem, **result, "correct": correct, "has_options": bool(problem.get("options"))}
                rows.append(row)
                writer.writerow([problem["id"], problem["language"], problem["level"], row["has_options"],
                                 problem["label"], result["answer"], int(correct), result["seconds"],
                                 result["status"], result["prompt_tokens"], result["retrieved"]])
                reasoning_out.write(json.dumps({
                    "id": problem["id"],
                    "language": problem["language"],
                    "answer": result["answer"],
                    "status": result["status"],
                    "seconds": result["seconds"],
                    "reasoning": result["reasoning"],
                }, ensure_ascii=False) + "\n")
                out.flush()
                reasoning_out.flush()
                running = 100 * sum(r["correct"] for r in rows) / len(rows)
                print(f"[{n}/{len(problems)}] {problem['id']} {problem['language']}: "
                      f"{result['answer']} (truth {problem['label']}) {result['seconds']:.1f} s, "
                      f"running accuracy {running:.1f}%", flush=True)

    hits = [r["correct"] for r in rows]
    seconds = [r["seconds"] for r in rows]
    floor = sum(r["label"] == CONFIG["FALLBACK_ANSWER"] for r in rows)
    over = sum(s > TIME_LIMIT for s in seconds)
    accuracy = 100 * sum(hits) / len(hits)

    print("\n" + "=" * 60)
    print(f"Vision reasoning log: {reasoning_path}")
    print(f"Vision crops: {crop_dir}")
    print(f"Accuracy: {sum(hits)}/{len(hits)} = {accuracy:.1f}%")
    print(f"  floor, always {CONFIG['FALLBACK_ANSWER']}: {100 * floor / len(rows):.1f}%   random guess: 20.0%")
    print(f"Latency: mean {statistics.mean(seconds):.1f} s, median {statistics.median(seconds):.1f} s, "
          f"worst {max(seconds):.1f} s, over {TIME_LIMIT:.0f} s: {over}")
    print(f"Status: {dict(Counter(r['status'] for r in rows))}")

    for field, title in (("language", "By language"), ("level", "By level"),
                         ("has_options", "By whether OCR found the options")):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r["correct"])
        table(title, groups)

    predicted = Counter(r["answer"] for r in rows)
    truth = Counter(r["label"] for r in rows)
    print("\nLetters   " + "  ".join(f"{l}:{predicted[l]:>3}/{truth[l]:<3}" for l in "ABCDE"))
    print("          (predicted/true; a model that loves one letter shows up here)")

    log = "results/ablations.csv"
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "name", "commit", "input", "languages", "n", "rag_k",
                             "accuracy", "mean_s", "worst_s", "over_limit"])
        writer.writerow([stamp, name, git_commit(), args.input, args.languages or "all", len(rows),
                         args.rag, f"{accuracy:.1f}", f"{statistics.mean(seconds):.1f}",
                         f"{max(seconds):.1f}", over])
    print(f"\nDetails: {detail_path}\nAblation log: {log}")


if __name__ == "__main__":
    main()
