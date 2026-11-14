#!/usr/bin/env python3
"""Analyse an eval.py CSV by topic, language, level and OCR option availability.

Example:
  python3 analyze_errors.py --eval results/eval-20261114-120000-baseline.csv \\
      --data data/val_topics.jsonl --output-dir results/analysis-baseline

If the data file has no topic field, this script applies the same first-pass
rules as classify_topics.py. Review low-confidence topics before reporting.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from copilot.topics import classify_problem


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_group_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dimension", "group", "correct", "total", "accuracy_percent", "error_percent"])
        for dimension, group, correct, total in rows:
            acc = 100.0 * correct / total if total else 0.0
            writer.writerow([dimension, group, correct, total, f"{acc:.2f}", f"{100-acc:.2f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, help="one results/eval-*.csv file from eval.py")
    parser.add_argument("--data", default="data/val_topics.jsonl", help="matching validation JSONL")
    parser.add_argument("--output-dir", default="results/error-analysis")
    parser.add_argument("--mistakes", type=int, default=10, help="number of wrong examples to export")
    args = parser.parse_args()

    problems = load_jsonl(args.data)
    by_key = {}
    for problem in problems:
        tagged = classify_problem(problem)
        by_key[(str(problem.get("id", "")), str(problem.get("language", "")))] = tagged

    with open(args.eval, newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if not predictions:
        parser.error("The evaluation CSV is empty")

    joined = []
    missing = 0
    for prediction in predictions:
        key = (str(prediction.get("id", "")), str(prediction.get("language", "")))
        problem = by_key.get(key)
        if problem is None:
            missing += 1
            continue
        try:
            is_correct = int(float(prediction.get("correct", "0"))) == 1
        except (TypeError, ValueError):
            is_correct = prediction.get("answer", "") == prediction.get("label", "")
        joined.append({**prediction, **problem, "correct_bool": is_correct})
    if not joined:
        parser.error("No matching (id, language) rows between --eval and --data")

    dimensions = {
        "topic": lambda r: r.get("topic", "other"),
        "language": lambda r: r.get("language", "unknown"),
        "level": lambda r: r.get("level", "unknown"),
        "has_options": lambda r: str(r.get("has_options", bool(r.get("options")))).lower(),
    }
    summaries = []
    for dimension, key_fn in dimensions.items():
        groups = defaultdict(list)
        for row in joined:
            groups[str(key_fn(row))].append(row["correct_bool"])
        for group, outcomes in sorted(groups.items()):
            summaries.append((dimension, group, sum(outcomes), len(outcomes)))

    os.makedirs(args.output_dir, exist_ok=True)
    group_path = os.path.join(args.output_dir, "groups.csv")
    write_group_csv(group_path, summaries)

    mistakes = [r for r in joined if not r["correct_bool"]]
    mistake_path = os.path.join(args.output_dir, "mistakes.csv")
    with open(mistake_path, "w", newline="", encoding="utf-8") as stream:
        fields = ["id", "language", "level", "topic", "topic_confidence", "label", "answer",
                  "has_options", "seconds", "status", "text"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in mistakes[:max(0, args.mistakes)]:
            writer.writerow(row)

    total = len(joined)
    correct = sum(r["correct_bool"] for r in joined)
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("# Error analysis\n\n")
        stream.write(f"- Evaluation file: `{os.path.basename(args.eval)}`\n")
        stream.write(f"- Validation data: `{os.path.basename(args.data)}`\n")
        stream.write(f"- Matched rows: **{total}** (unmatched: {missing})\n")
        stream.write(f"- Accuracy: **{correct}/{total} = {100*correct/total:.2f}%**\n\n")
        for dimension in dimensions:
            stream.write(f"## By {dimension.replace('_', ' ')}\n\n")
            stream.write("| Group | Correct / total | Accuracy | Error |\n|---|---:|---:|---:|\n")
            for dim, group, hits, n in summaries:
                if dim == dimension:
                    acc = 100 * hits / n
                    stream.write(f"| {group} | {hits}/{n} | {acc:.1f}% | {100-acc:.1f}% |\n")
            stream.write("\n")
        stream.write("## Manual review\n\n")
        stream.write(f"Inspect the first {min(len(mistakes), max(0,args.mistakes))} wrong answers in `mistakes.csv` one by one. "
                     "Topic labels with low confidence should be checked before drawing conclusions.\n")

    print(f"Matched {total} predictions; accuracy {correct}/{total} = {100*correct/total:.2f}%")
    print(f"Groups:   {group_path}")
    print(f"Mistakes: {mistake_path}")
    print(f"Report:   {report_path}")
    if missing:
        print(f"Warning: {missing} prediction rows had no matching problem in {args.data}")


if __name__ == "__main__":
    main()
