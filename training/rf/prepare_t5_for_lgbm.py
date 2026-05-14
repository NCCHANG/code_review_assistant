"""Convert T5 training data (CTSSB-1M) into additional rows for LightGBM training.

Maps each sstub_pattern to one of the four LightGBM labels so the gate model
learns to detect the same bug types that T5 is trained to fix.

Mapping rationale:
  Label 1 – Wrong Binary Operator : wrong operator/operand in an expression
  Label 2 – Variable Misuse       : wrong identifier, attribute, literal, or call name
  Label 3 – Swapped Operand       : wrong argument count/order or structural addition

Output:
    training/rf/data/t5_augment.csv   (columns: function, label, bug_type, source_task)

Usage:
    python training/rf/prepare_t5_for_lgbm.py
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T5_TRAIN_CSV = os.path.join(SCRIPT_DIR, "..", "t5", "data", "train.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "data", "t5_augment.csv")

# Map every CTSSB sstub_pattern to a LightGBM label (0 = Clean, never used here)
SSTUB_TO_LABEL = {
    # ── Label 1: Wrong Binary Operator ──────────────────────────────────────
    "CHANGE_BINARY_OPERATOR":        1,
    "CHANGE_BINARY_OPERAND":         1,
    "CHANGE_UNARY_OPERATOR":         1,
    "CHANGE_CONSTANT_TYPE":          1,

    # ── Label 2: Variable Misuse ─────────────────────────────────────────────
    "CHANGE_IDENTIFIER_USED":        2,
    "CHANGE_ATTRIBUTE_USED":         2,
    "WRONG_FUNCTION_NAME":           2,
    "CHANGE_STRING_LITERAL":         2,
    "CHANGE_NUMERIC_LITERAL":        2,
    "CHANGE_BOOLEAN_LITERAL":        2,
    "CHANGE_KEYWORD_ARGUMENT_USED":  2,
    "SAME_FUNCTION_WRONG_CALLER":    2,
    "SINGLE_TOKEN":                  2,
    "MORE_SPECIFIC_IF":              2,
    "LESS_SPECIFIC_IF":              2,

    # ── Label 3: Swapped / Structural Operand ────────────────────────────────
    "SAME_FUNCTION_MORE_ARGS":           3,
    "SAME_FUNCTION_LESS_ARGS":           3,
    "SAME_FUNCTION_SWAP_ARGS":           3,
    "ADD_FUNCTION_AROUND_EXPRESSION":    3,
    "ADD_METHOD_CALL":                   3,
    "ADD_ELEMENTS_TO_ITERABLE":          3,
    "ADD_ATTRIBUTE_ACCESS":              3,
}

LABEL_NAMES = {
    1: "Wrong Binary Operator",
    2: "Variable Misuse",
    3: "Swapped Operand",
}


def main():
    if not os.path.exists(T5_TRAIN_CSV):
        print(f"Error: {T5_TRAIN_CSV} not found.")
        sys.exit(1)

    t5 = pd.read_csv(T5_TRAIN_CSV)
    print(f"Loaded {len(t5):,} rows from T5 train.csv")

    # Map patterns → labels; drop any pattern not in the dict
    t5["label"] = t5["sstub_pattern"].map(SSTUB_TO_LABEL)
    unmapped = t5["label"].isna().sum()
    if unmapped:
        print(f"  Dropped {unmapped:,} rows with unmapped patterns.")
    t5 = t5.dropna(subset=["label"])
    t5["label"] = t5["label"].astype(int)

    # Rename input_text → function to match LightGBM CSV schema
    out = pd.DataFrame({
        "function":    t5["input_text"].str.strip(),
        "label":       t5["label"],
        "bug_type":    t5["label"].map(LABEL_NAMES),
        "source_task": "ctssb_t5",
    })

    # Remove empty snippets
    out = out[out["function"].str.len() > 0].reset_index(drop=True)

    print(f"Final augmentation rows: {len(out):,}")
    print("\nLabel distribution:")
    for lbl, cnt in out["label"].value_counts().sort_index().items():
        print(f"  Label {lbl} ({LABEL_NAMES[lbl]}): {cnt:,}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print("Re-run train_rf.py to retrain LightGBM with augmented data.")


if __name__ == "__main__":
    main()
