import pandas as pd
from sklearn.model_selection import train_test_split

# 1. LOAD YOUR MASTER DATA
try:
    df = pd.read_csv("synthetic_python_bugs.csv")
    print(f"Loaded Master Dataset: {len(df)} pairs")
except FileNotFoundError:
    # Fallback dummy data if you haven't run the previous script yet
    print("Warning: 'synthetic_python_bugs.csv' not found. Creating dummy data.")
    data = [
        {"buggy_code": "def f(x): return x - 1", "fixed_code": "def f(x): return x + 1"},
        {"buggy_code": "if x = 1:", "fixed_code": "if x == 1:"},
        {"buggy_code": "print x", "fixed_code": "print(x)"},
        {"buggy_code": "import nmp as np", "fixed_code": "import numpy as np"}
    ] * 20 # duplicate to make it big enough to split
    df = pd.DataFrame(data)

# 2. SPLIT THE MASTER DATA FIRST (Crucial Step!)
# We split the PAIRS. This ensures a specific bug never leaks from Train to Test.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"\nSplitting Master Data:")
print(f"Train Pairs: {len(train_df)}")
print(f"Test Pairs:  {len(test_df)}")

# --- 3. PREPARE FOR RANDOM FOREST (Classifier) ---
# Goal: Columns ['code_snippet', 'label'] (0 or 1)

def prepare_for_classifier(dataframe):
    # Take the Buggy ones (Label 1)
    buggy = dataframe[['buggy_code']].copy()
    buggy.columns = ['code_snippet']
    buggy['label'] = 1
    
    # Take the Fixed ones (Label 0)
    fixed = dataframe[['fixed_code']].copy()
    fixed.columns = ['code_snippet']
    fixed['label'] = 0
    
    # Combine them
    combined = pd.concat([buggy, fixed])
    # Shuffle them so 0s and 1s are mixed
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)

rf_train = prepare_for_classifier(train_df)
rf_test  = prepare_for_classifier(test_df)

# Save Files
rf_train.to_csv("rf_train_dataset.csv", index=False)
rf_test.to_csv("rf_test_dataset.csv", index=False)
print("\n[Random Forest] Created 'rf_train_dataset.csv' & 'rf_test_dataset.csv'")
print(f"   - Training Samples: {len(rf_train)} (Half buggy, half clean)")

# --- 4. PREPARE FOR CODET5 (Repairer) ---
# Goal: Columns ['input_text', 'target_text']
# We only want to teach it to fix BUGGY code. 
# (We don't need to show it perfect code, it just needs to fix errors)

def prepare_for_codet5(dataframe):
    # Just rename columns to standard huggingface format
    new_df = dataframe.rename(columns={'buggy_code': 'input_text', 'fixed_code': 'target_text'})
    return new_df

t5_train = prepare_for_codet5(train_df)
t5_test  = prepare_for_codet5(test_df)

# Save Files
t5_train.to_csv("t5_train_dataset.csv", index=False)
t5_test.to_csv("t5_test_dataset.csv", index=False)
print("\n[CodeT5] Created 't5_train_dataset.csv' & 't5_test_dataset.csv'")
print(f"   - Training Pairs: {len(t5_train)}")