#!/usr/bin/env python3
"""Fast OCR for step 9: uses all CPU cores and fills data/.ocr_cache.jsonl.

prepare_m3kang.py does the OCR one image at a time, on one process. This
script does the same OCR (same parser, same cache file) with several worker
processes. Run it BEFORE prepare_m3kang.py; afterwards prepare_m3kang.py finds
everything in the cache and finishes in a minute.

    python laptop/ocr_parallel.py --languages eng,cat
    python laptop/prepare_m3kang.py --languages eng,cat --save-images

If you stop it (Ctrl+C), run it again: it skips what is already cached.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ocr = None


def init_worker():
    global _ocr
    from rapidocr_onnxruntime import RapidOCR
    try:  # one thread per worker: the parallelism comes from the processes
        _ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    except TypeError:
        _ocr = RapidOCR()


def work(task):
    from prepare_m3kang import parse_options
    key, image = task
    boxes, _ = _ocr(image)
    return key, parse_options(boxes or [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", default="eng,cat")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="data")
    p.add_argument("--parquet", nargs="*", help="local parquet files instead of downloading shard 0")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import pyarrow.parquet as pq
    from prepare_m3kang import DATASET, SHARD, SHARDS
    paths = args.parquet
    if not paths:
        from huggingface_hub import hf_hub_download
        name = SHARD.format(split="high_quality", index=0, total=SHARDS["high_quality"])
        paths = [hf_hub_download(DATASET, name, repo_type="dataset")]

    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, ".ocr_cache.jsonl")
    done = set()
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}

    languages = set(args.languages.split(","))
    tasks = []
    for path in paths:
        table = pq.read_table(path, columns=["id", "language", "image"])
        ids, langs = table.column("id").to_pylist(), table.column("language").to_pylist()
        for i, (id_, lang) in enumerate(zip(ids, langs)):
            key = f"{id_}|{lang}"
            if lang in languages and key not in done:
                tasks.append((key, table.column("image")[i].as_py()["bytes"]))
        del table
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} images to read ({len(done)} already cached), {args.workers} workers", flush=True)
    if not tasks:
        return

    t0, n = time.time(), 0
    with Pool(args.workers, initializer=init_worker) as pool, open(cache_path, "a", encoding="utf-8") as cache:
        for key, options in pool.imap_unordered(work, tasks, chunksize=4):
            cache.write(json.dumps({"key": key, "options": options}, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                cache.flush()
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(tasks)}  {rate:.1f} images/s  ~{(len(tasks) - n) / rate / 60:.0f} min left", flush=True)
    print("Done. Now run: python laptop/prepare_m3kang.py --languages", args.languages, "--save-images")


if __name__ == "__main__":
    main()
