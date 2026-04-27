"""Stratified train/test split for the merged CuBERT dataset.

Reads data/merged_cubert_dataset.csv and writes data/train.csv and
data/test.csv with proportional representation of all four classes.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "merged_cubert_dataset.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_NAMES = {0: "Clean", 1: "Wrong Binary Operator", 2: "Variable Misuse", 3: "Swapped Operand"}


def print_distribution(df, title):
    """Print per-class counts and percentages."""
    total = len(df)
    print(f"\n{title} ({total:,} samples):")
    for label in sorted(df["label"].unique()):
        count = (df["label"] == label).sum()
        pct = 100 * count / total
        print(f"  Label {label} ({LABEL_NAMES[label]}): {count:,} ({pct:.1f}%)")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Run download_and_preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records from {INPUT_FILE}")
    print_distribution(df, "Full dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    print_distribution(train_df, "Train split")
    print_distribution(test_df, "Test split")

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\nSaved: {TRAIN_FILE} ({len(train_df):,} rows)")
    print(f"Saved: {TEST_FILE}  ({len(test_df):,} rows)")


if __name__ == "__main__":
    main()
