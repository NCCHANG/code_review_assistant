import os
import sys
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

script_dir = os.path.dirname(os.path.abspath(__file__))
# use trained model else base model for testing
trained_model_path = os.path.join(script_dir, "training", "t5", "saved_model")
base_model_name = "Salesforce/codet5-base"

if os.path.exists(trained_model_path):
    print(f"Loading Trained Model from: {trained_model_path}")
    load_path = trained_model_path
else:
    print(f"Trained model not found (saved_model).")
    print(f"Loading Base Model for testing: {base_model_name}")
    load_path = base_model_name

try:
    tokenizer = RobertaTokenizer.from_pretrained(load_path)
    model = T5ForConditionalGeneration.from_pretrained(load_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def fix_bug(buggy_code, intention=""):
    # Prefix as used in training
    if intention != "":
        input_text = f"fix intent: {str(intention).strip()} code: {buggy_code}"
    else:
        input_text = "fix: " + buggy_code
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate
    outputs = model.generate(inputs, max_length=128)
    
    # Decode
    fixed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fixed_code

if __name__ == "__main__":
    code = "def add(a, b):\n    return a - b"
    fixed = fix_bug(code)
    print(f"Fixed Code >> {fixed}")
