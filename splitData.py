import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_and_split_data():
    print("🚀 Loading the massive local dataset...")
    # Load the data you just mined
    df = pd.DataFrame(pd.read_csv("massive_local_bugs_dataset.csv"))
    
    # Drop any accidental duplicates or empty rows
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"✅ Cleaned dataset contains {len(df)} unique bug/fix pairs.")

    # Create directories to store the formatted data
    os.makedirs("./processed_data/t5", exist_ok=True)
    os.makedirs("./processed_data/rf", exist_ok=True)

    # ==========================================
    # 1. PREPARE CODET5 DATA (80% Train, 20% Test)
    # ==========================================
    print("\n📦 Splitting data for CodeT5 (Repairer)...")
    
    # NEW: Combine the commit message (intent) with the buggy code
    # We add a safety check .fillna('') just in case a commit message was blank
    df['input_text'] = "Intent: " + df['commit_message'].fillna('') + " Code:\n" + df['input_text']
    
    # T5 now uses the combined intent+code as input, and the fix as target
    t5_df = df[['input_text', 'target_text']]
    
    t5_train, t5_test = train_test_split(t5_df, test_size=0.20, random_state=42)
    
    t5_train.to_csv("./processed_data/t5/t5_train.csv", index=False)
    t5_test.to_csv("./processed_data/t5/t5_test.csv", index=False)
    print(f"   -> CodeT5 Train: {len(t5_train)} pairs")
    print(f"   -> CodeT5 Test: {len(t5_test)} pairs")


    # ==========================================
    # 2. PREPARE RANDOM FOREST DATA (Buggy=1, Clean=0)
    # ==========================================
    print("\n📦 Splitting data for Random Forest (Classifier)...")
    
    # Create Buggy rows (Label = 1)
    buggy_df = pd.DataFrame({
        'code': df['input_text'],
        'is_buggy': 1
    })
    
    # Create Clean rows (Label = 0)
    clean_df = pd.DataFrame({
        'code': df['target_text'],
        'is_buggy': 0
    })
    
    # Combine them into one big classification dataset
    rf_df = pd.concat([buggy_df, clean_df], ignore_index=True)
    
    # Shuffle the dataset so buggy and clean are mixed up
    rf_df = rf_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    rf_train, rf_test = train_test_split(rf_df, test_size=0.20, random_state=42)
    
    rf_train.to_csv("./processed_data/rf/rf_train.csv", index=False)
    rf_test.to_csv("./processed_data/rf/rf_test.csv", index=False)
    print(f"   -> RF Train: {len(rf_train)} snippets")
    print(f"   -> RF Test: {len(rf_test)} snippets")

    print("\n🎉 ALL DATA PREPARED SUCCESSFULLY!")
    print("You are now ready to run your training scripts.")

if __name__ == "__main__":
    process_and_split_data()