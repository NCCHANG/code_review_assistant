import os
import sys
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from openai import OpenAI
from google import genai

#Setting UP Grok OpenAI Client
GROQ_KEY = "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" 
groq_client = OpenAI(
    api_key=GROQ_KEY,
    base_url="https://api.groq.com/openai/v1"
)
#Setting up gemini GenAI Client
GEMINI_KEY = "AIzaSyCdkseSms9ughYFLw0OswbW6P9_wg3Cy70"
gemini_client = genai.Client(api_key=GEMINI_KEY)

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

def fix_bug(buggy_code: str, intention=""):
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

def generate_feedback(buggy_code: str, fixed_code: str):
    prompt = f"""
    You are a Code Reviewer.
    Buggy Code: {buggy_code}
    Fixed Code: {fixed_code}
    
    Task:
    1. Analysis: 1 sentence on the bug.
    2. Reasoning: 1 or 2 sentences on the fix.
    
    Format:
    Analysis: ...
    Reasoning: ...
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"\n[!] Gemini Failed. Switching to Groq...")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": "You are a helpful code reviewer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n[!] Groq Failed ({e}). Failed to generate feedback.")

if __name__ == "__main__":
    code = "def add(a, b):\n    return a - b"
    fixed = fix_bug(code)
    feedback = generate_feedback(code, fixed)
    print("\n--- FEEDBACK ---")
    print(feedback)
    print("\n--- FIXED CODE ---")
    print(f"Fixed Code >> {fixed}")
