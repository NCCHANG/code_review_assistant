"""Download and preprocess the CuBERT ETH Py150 bug-detection dataset.

Loads three tasks from HuggingFace, maps string labels to integers, deduplicates
clean examples so they don't outweigh buggy ones 3-to-1, then saves a single
merged CSV to data/merged_cubert_dataset.csv.
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_NAME = "claudios/cubert_ETHPy150Open"

# HuggingFace config names (note the _datasets suffix) and their label mappings.
# "Correct" always maps to 0; the bug label is task-specific.
TASKS = {
    "wrong_binary_operator_datasets": {
        "bug_type": "Wrong Binary Operator",
        "label_map": {"Correct": 0, "Wrong binary operator": 1},
    },
    "variable_misuse_datasets": {
        "bug_type": "Variable Misuse",
        "label_map": {"Correct": 0, "Variable misuse": 2},
    },
    "swapped_operands_datasets": {
        "bug_type": "Swapped Operand",
        "label_map": {"Correct": 0, "Swapped operands": 3},
    },
}

SPLITS = ("train", "dev", "test")

LABEL_NAMES = {0: "Clean", 1: "Wrong Binary Operator", 2: "Variable Misuse", 3: "Swapped Operand"}

# Stratified sample target. Set to None to keep the full deduplicated dataset.
TARGET_SAMPLES = 175_000
RANDOM_STATE = 42

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_cubert_dataset.csv")


def load_task_splits(task_name):
    """Load train/dev/test splits for one task and return a flat list of records."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    records = []
    for split in SPLITS:
        try:
            ds = load_dataset(DATASET_NAME, task_name, split=split, trust_remote_code=True)
            records.extend(ds.to_list())
            print(f"  [{task_name}] {split}: {len(ds):,} records")
        except Exception as exc:
            print(f"  Warning: could not load {task_name}/{split}: {exc}")
    return records


def build_task_dataframe(records, task_name, task_config):
    """Convert raw HuggingFace records for one task into a tidy DataFrame."""
    label_map = task_config["label_map"]
    bug_type = task_config["bug_type"]
    rows = []
    skipped = 0
    for rec in records:
        label_str = rec.get("label", "")
        if label_str not in label_map:
            skipped += 1
            continue
        label_int = label_map[label_str]
        rows.append({
            "function": rec["function"],
            "label": label_int,
            "bug_type": "Clean" if label_int == 0 else bug_type,
            "source_task": task_name,
        })
    if skipped:
        print(f"  [{task_name}] Skipped {skipped} records with unrecognised labels")
    return pd.DataFrame(rows)


def deduplicate_clean_examples(df):
    """Drop duplicate clean (label=0) rows across tasks using function text as key.

    Without this step, the same clean function can appear once per task,
    making label-0 examples 3x more frequent than each buggy class.
    """
    clean = df[df["label"] == 0].drop_duplicates(subset=["function"])
    buggy = df[df["label"] != 0]
    return pd.concat([clean, buggy], ignore_index=True)


def stratified_sample(df, target_n, random_state=42):
    """Proportionally sample target_n rows from df, preserving class ratios.

    Each class is sampled at the same fraction so the label distribution is
    maintained. If a class has fewer rows than its quota, all its rows are kept.
    """
    fraction = target_n / len(df)
    parts = []
    for _, group in df.groupby("label"):
        n = min(len(group), max(1, round(len(group) * fraction)))
        parts.append(group.sample(n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def print_distribution(df, title=""):
    if title:
        print(f"\n{title}:")
    total = len(df)
    for label in sorted(df["label"].unique()):
        count = (df["label"] == label).sum()
        pct = 100 * count / total
        print(f"  Label {label} ({LABEL_NAMES[label]}): {count:,} ({pct:.1f}%)")
    print(f"  Total: {total:,}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_dfs = []
    for task_name, task_config in TASKS.items():
        print(f"\nLoading: {task_name}")
        records = load_task_splits(task_name)
        df = build_task_dataframe(records, task_name, task_config)
        print(f"  -> {len(df):,} usable records")
        task_dfs.append(df)

    merged = pd.concat(task_dfs, ignore_index=True)
    print_distribution(merged, "Before deduplication")

    merged = deduplicate_clean_examples(merged)
    print_distribution(merged, "After deduplication (clean examples deduplicated)")

    if TARGET_SAMPLES is not None:
        merged = stratified_sample(merged, TARGET_SAMPLES, random_state=RANDOM_STATE)
        print_distribution(merged, f"After stratified sampling (target={TARGET_SAMPLES:,})")

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
