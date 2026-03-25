import pandas as pd
import sys
from sklearn.model_selection import train_test_split

# Load dataset
try:
    df = pd.read_csv("synthetic_python_bugs.csv")
    print(f"Loaded Master Dataset: {len(df)} pairs")
except FileNotFoundError:
    sys.exit(-1)

# Splitting dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train Pairs: {len(train_df)}")
print(f"Test Pairs:  {len(test_df)}")

# PREPARE DATASET FOR CLASSIFIER
# to have columns ['code_snippet', 'label'] where label is 1 for buggy and 0 for clean code

def prepare_for_classifier(dataframe):
    # Take buggy one from dataset and label as 1
    buggy = dataframe[['buggy_code']].copy()
    buggy.columns = ['code_snippet']
    buggy['label'] = 1
    
    # Take non-buggy one from dataset and label as 0
    fixed = dataframe[['fixed_code']].copy()
    fixed.columns = ['code_snippet']
    fixed['label'] = 0
    
    # Combine both safe and non safe
    combined = pd.concat([buggy, fixed])
    # and shuffle
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)

rf_train = prepare_for_classifier(train_df)
rf_test  = prepare_for_classifier(test_df)

# Save Files
rf_train.to_csv("rf_train_dataset.csv", index=False)
rf_test.to_csv("rf_test_dataset.csv", index=False)
print("\n[Random Forest] Created 'rf_train_dataset.csv' & 'rf_test_dataset.csv'")
print(f"   - Training Samples: {len(rf_train)} (Half buggy, half clean)")

# NOW FOR CODET5................
# Goal: Columns ['input_text', 'target_text']
# This is to make it treat code fixing as a translation task, where input is the buggy code and output is the fixed code.

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