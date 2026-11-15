#!/usr/bin/env python3
"""Pick the best results/eval-*.csv run and the commit behind it.

    python3 pick_best.py                # scan results/
    python3 pick_best.py --dir results  # same, explicit

Reads every results/eval-*.csv, scores accuracy and latency, matches each run
to its commit through results/ablations.csv (where eval.py logs it), ranks
both the runs and the commits, and prints the git command to branch the
winning state:

    git checkout -b best-<name> <commit>
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys

NAME_RE = re.compile(r"^eval-(\d{8})-(\d{6})-(.+)\.csv$")


def load_commits(log_path):
    """Map (time, name) -> commit from the ablation log."""
    out = {}
    if not os.path.exists(log_path):
        return out
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            commit = (row.get("commit") or "").strip()
            if commit:
                out[(row.get("time", "").strip(), row.get("name", "").strip())] = commit
    return out


def score_eval(path):
    """Return stats for one eval-*.csv, or None if it can't be read."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        print(f"warning: can't read {path}: {e}", file=sys.stderr)
        return None
    rows = [r for r in rows if r.get("id")]
    if not rows:
        return None

    def num(r, key, default=0.0):
        try:
            return float(r[key])
        except (KeyError, TypeError, ValueError):
            return default

    correct = [r.get("correct", "").strip() in ("1", "true", "True") for r in rows]
    seconds = [num(r, "seconds") for r in rows]
    bad = sum(r.get("status", "").strip().lower().startswith(("error", "timeout")) for r in rows)
    return {
        "path": path,
        "name": os.path.basename(path),
        "n": len(rows),
        "accuracy": 100.0 * sum(correct) / len(rows),
        "hits": sum(correct),
        "mean_s": statistics.mean(seconds),
        "worst_s": max(seconds),
        "bad": bad,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results", help="directory holding eval-*.csv")
    args = p.parse_args()

    runs = [s for s in (score_eval(f) for f in sorted(glob.glob(os.path.join(args.dir, "eval-*.csv"))))
            if s]
    if not runs:
        sys.exit(f"no eval-*.csv files found in {args.dir}/")

    commits = load_commits(os.path.join(args.dir, "ablations.csv"))
    for s in runs:
        m = NAME_RE.match(s["name"])
        key = (f"{m.group(1)}-{m.group(2)}", m.group(3)) if m else ("", s["name"][5:-4])
        s["commit"] = commits.get(key, "")
        s["label"] = key[1]

    runs.sort(key=lambda s: (-s["accuracy"], s["mean_s"], s["bad"]))

    print(f"{'rank':>4}  {'acc':>6}  {'n':>4}  {'mean_s':>7}  {'worst_s':>7}  {'bad':>3}  "
          f"{'commit':<9}  run")
    for i, s in enumerate(runs, 1):
        print(f"{i:>4}  {s['accuracy']:>5.1f}%  {s['n']:>4}  {s['mean_s']:>7.1f}  "
              f"{s['worst_s']:>7.1f}  {s['bad']:>3}  {s['commit'] or '?':<9}  {s['name']}")

    # Best run per commit, ranked: the state you would branch from.
    best_by_commit = {}
    for s in runs:  # runs already sorted best-first
        if s["commit"] and s["commit"] not in best_by_commit:
            best_by_commit[s["commit"]] = s
    print("\nBest run per commit:")
    for i, (commit, s) in enumerate(sorted(best_by_commit.items(),
                                           key=lambda kv: (-kv[1]["accuracy"], kv[1]["mean_s"])), 1):
        print(f"  {i}. {commit}  {s['accuracy']:.1f}% ({s['hits']}/{s['n']}, "
              f"mean {s['mean_s']:.1f} s)  <- {s['name']}")

    best = runs[0]
    print("\n" + "=" * 70)
    if best["commit"]:
        branch = re.sub(r"[^A-Za-z0-9._-]+", "-", best["label"]).strip("-") or "best"
        print(f"Winner: {best['name']}")
        print(f"  {best['accuracy']:.1f}% ({best['hits']}/{best['n']}), mean {best['mean_s']:.1f} s, "
              f"worst {best['worst_s']:.1f} s, {best['bad']} bad status")
        print(f"Commit: {best['commit']}")
        print(f"\n  git checkout -b best-{branch} {best['commit']}")
    else:
        print(f"Winner by numbers: {best['name']} ({best['accuracy']:.1f}%)")
        print("No commit found for it in results/ablations.csv; check manually with git log.")


if __name__ == "__main__":
    main()
