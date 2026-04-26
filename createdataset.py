"""
Training dataset: TSSB-3M (Richter & Wehrheim, MSR 2022)
  Source: https://zenodo.org/records/5845439
  Format: jsonlines inside tssb_data_3M.zip
  Key fields: before, after, sstub_pattern, likely_bug, in_function

  Strategy:
  - Download zip once (~912 MB), stored in ./datasets/
  - Stream jsonlines from inside the zip (no full extraction needed)
  - Filter: likely_bug=True, in_function=True, sstub_pattern != SINGLE_STMT
  - Fill per-class buckets until each has TARGET_PER_CLASS rows
  - Balanced sample: every class gets exactly min(target, smallest_class) rows
  - commit_message = human-readable version of sstub_pattern (used as T5 intention)

Run: .venv/bin/python createdataset.py
"""

import gzip
import io
import json
import os
import random
import zipfile
from collections import defaultdict

import pandas as pd
import requests
from tqdm import tqdm

TARGET_PER_CLASS = 3_000
MAX_STREAM       = 3_000_000    # stream full dataset to fill all classes
ZENODO_URL       = "https://zenodo.org/records/5845439/files/tssb_data_3M.zip"
ZIP_PATH         = "./datasets/tssb_data_3M.zip"
OUTPUT_PATH      = "./tssb3m_stratified.csv"

# Human-readable labels for T5 intention field (used instead of commit messages)
PATTERN_TO_INTENTION = {
    "CHANGE_IDENTIFIER":              "fix wrong variable name",
    "CHANGE_BINARY_OPERATOR":         "fix wrong binary operator",
    "CHANGE_UNARY_OPERATOR":          "fix wrong unary operator",
    "CHANGE_NUMERIC_LITERAL":         "fix wrong numeric literal",
    "CHANGE_STRING_LITERAL":          "fix wrong string literal",
    "CHANGE_BOOLEAN_LITERAL":         "fix wrong boolean literal",
    "CHANGE_ATTRIBUTE_USED":          "fix wrong attribute access",
    "CHANGE_CALLER_IN_FUNCTION_CALL": "fix wrong function being called",
    "CHANGE_FUNCTION_CALL":           "fix wrong function call",
    "ADD_FUNCTION_AROUND_EXPRESSION": "wrap expression in missing function call",
    "ADD_ARGUMENTS":                  "add missing argument to function call",
    "REMOVE_ARGUMENTS":               "remove extra argument from function call",
    "SWAP_ARGUMENTS":                 "swap incorrectly ordered arguments",
    "MORE_SPECIFIC_IF":               "tighten if condition",
    "LESS_SPECIFIC_IF":               "loosen if condition",
    "MISSING_ELSE_BRANCH":            "add missing else branch",
}


def download_zip():
    if os.path.exists(ZIP_PATH):
        print(f"Dataset zip already present at {ZIP_PATH} — skipping download.")
        return

    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    print(f"Downloading TSSB-3M from Zenodo (~912 MB) …")
    print(f"URL: {ZENODO_URL}")

    resp = requests.get(ZENODO_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(ZIP_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in resp.iter_content(chunk_size=65_536):
            f.write(chunk)
            pbar.update(len(chunk))

    print("Download complete.")


def stream_jsonlines(zip_path: str):
    """Yield parsed JSON entries from all .jsonl.gz (or .jsonl) files inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        targets = [n for n in names if n.endswith(".jsonl.gz") or n.endswith(".jsonl") or n.endswith(".json")]
        if not targets:
            raise ValueError(f"No .jsonl/.jsonl.gz file found in zip. Contents: {names}")

        print(f"Found {len(targets)} shard(s) to stream …")

        for target in targets:
            with zf.open(target) as raw:
                stream = gzip.open(io.BufferedReader(raw)) if target.endswith(".gz") else raw
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line.decode("utf-8", errors="ignore"))
                    except json.JSONDecodeError:
                        continue


def fill_buckets(target_per_class: int) -> dict[str, list]:
    """
    Stream TSSB-3M and collect rows into per-SStuB-pattern buckets.
    Stops as soon as every discovered class has target_per_class rows,
    or MAX_STREAM total rows have been read.
    """
    buckets: dict[str, list] = defaultdict(list)
    total_seen = 0

    for entry in tqdm(stream_jsonlines(ZIP_PATH), desc="Streaming rows", total=MAX_STREAM):
        total_seen += 1
        if total_seen > MAX_STREAM:
            break

        # Apply quality filters
        if not entry.get("likely_bug"):
            continue
        if not entry.get("in_function"):
            continue

        pattern = entry.get("sstub_pattern", "")
        if not pattern or pattern == "SINGLE_STMT":
            continue

        if len(buckets[pattern]) < target_per_class:
            buckets[pattern].append(entry)

        # Stop early once every known class is full
        if len(buckets) >= 2 and all(len(v) >= target_per_class for v in buckets.values()):
            print(f"\nAll {len(buckets)} classes full after {total_seen:,} rows — stopping early.")
            break

    return buckets


def balanced_sample(buckets: dict[str, list]) -> pd.DataFrame:
    """
    Sample exactly min(TARGET_PER_CLASS, smallest_bucket) rows per class.
    Every class ends up with the same count — guaranteed.
    """
    min_count = min(len(v) for v in buckets.values())
    per_class  = min(TARGET_PER_CLASS, min_count)

    print(f"\nClass sizes before sampling:")
    for label in sorted(buckets):
        print(f"  {label:<45} {len(buckets[label]):>6} rows")
    print(f"\nSampling exactly {per_class} per class (smallest class = {min_count}).")

    random.seed(42)
    records = []
    for pattern, rows in buckets.items():
        intention = PATTERN_TO_INTENTION.get(pattern, pattern.lower().replace("_", " "))
        for entry in random.sample(rows, per_class):
            records.append({
                "input_text":     entry.get("before", ""),
                "target_text":    entry.get("after", ""),
                "bug_type":       pattern,
                "commit_message": intention,   # used as T5 intention at training time
                "repo":           entry.get("project", "TSSB3M"),
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    download_zip()

    buckets = fill_buckets(TARGET_PER_CLASS)

    if not buckets:
        print("ERROR: No labeled rows collected. Check filters in fill_buckets().")
    else:
        df = balanced_sample(buckets)

        df = df.dropna(subset=["input_text", "target_text"])
        df = df[df["input_text"].str.strip() != df["target_text"].str.strip()]

        df.to_csv(OUTPUT_PATH, index=False)

        print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
        print(f"\nFinal distribution (should be equal):")
        print(df["bug_type"].value_counts().to_string())
