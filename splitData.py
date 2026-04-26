import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_and_split_data():
    print("🚀 Loading TSSB-3M stratified dataset...")
    df = pd.read_csv("tssb3m_stratified.csv")

    df = df.dropna(subset=["input_text", "target_text", "bug_type"])
    df = df.drop_duplicates()

    FOCUS_BUG_TYPES = [
        "CHANGE_BOOLEAN_LITERAL",
        "ADD_ELEMENTS_TO_ITERABLE",
        "CHANGE_BINARY_OPERAND",
    ]
    df = df[df['bug_type'].isin(FOCUS_BUG_TYPES)].reset_index(drop=True)
    print(f"✅ Filtered to {df['bug_type'].nunique()} bug types: {FOCUS_BUG_TYPES}")
    print(f"   {len(df)} unique pairs remaining.")

    # Create directories to store the formatted data
    os.makedirs("./processed_data/t5", exist_ok=True)
    os.makedirs("./processed_data/rf", exist_ok=True)

    # ==========================================
    # 1. PREPARE CODET5 DATA (80% Train, 20% Test)
    # ==========================================
    print("\n📦 Splitting data for CodeT5 (Repairer)...")
    
    # Keep commit_message as a separate intention column so t5_trainer can
    # format it as: "fix intent: <intention> code: <code>"
    df['intention'] = df['commit_message'].fillna('')

    t5_df = df[['input_text', 'target_text', 'intention']]
    
    t5_train, t5_test = train_test_split(t5_df, test_size=0.20, random_state=42)
    
    t5_train.to_csv("./processed_data/t5/t5_train.csv", index=False)
    t5_test.to_csv("./processed_data/t5/t5_test.csv", index=False)
    print(f"   -> CodeT5 Train: {len(t5_train)} pairs")
    print(f"   -> CodeT5 Test: {len(t5_test)} pairs")


    # ==========================================
    # 2. PREPARE RANDOM FOREST DATA (bug type + CLEAN binary)
    # ==========================================
    # RF classifies bug type on buggy code, plus a CLEAN class from fixed code.
    # CLEAN class uses target_text (post-fix code) sampled to match per-class count.
    print("\n📦 Splitting data for Random Forest (bug-type + clean classifier)...")

    buggy_df = pd.DataFrame({
        'code':     df['input_text'],
        'bug_type': df['bug_type'],
    })

    # Sample same number of clean examples as one bug class to keep balance
    per_class_count = len(df) // df['bug_type'].nunique()
    clean_df = pd.DataFrame({
        'code':     df['target_text'].sample(n=per_class_count, random_state=42).values,
        'bug_type': 'CLEAN',
    })

    rf_df = pd.concat([buggy_df, clean_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    rf_train, rf_test = train_test_split(
        rf_df, test_size=0.20, random_state=42, stratify=rf_df['bug_type']
    )

    rf_train.to_csv("./processed_data/rf/rf_train.csv", index=False)
    rf_test.to_csv("./processed_data/rf/rf_test.csv", index=False)
    print(f"   -> RF Train: {len(rf_train)} snippets")
    print(f"   -> RF Test:  {len(rf_test)} snippets")
    print(f"   -> Classes:  {sorted(rf_df['bug_type'].unique())}")

    print("\n🎉 ALL DATA PREPARED SUCCESSFULLY!")
    print("You are now ready to run your training scripts.")

if __name__ == "__main__":
    process_and_split_data()