import os
import sys
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from openai import OpenAI
from google import genai

class Repairer:
    GROQ_KEY = "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"
    groq_client = None
    GEMINI_KEY = "AIzaSyCdkseSms9ughYFLw0OswbW6P9_wg3Cy70"
    gemini_client = None
    def __init__(self):
        self._setup_AI_clients()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.trained_model_path = os.path.join(self.script_dir, "training", "t5", "saved_model")
        self._setup_model(self.trained_model_path)
        
    def _setup_AI_clients(self):
        #Setting UP Grok OpenAI Client
        try:
            self.groq_client = OpenAI(
                api_key=self.GROQ_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            print("Groq client initialized.")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.groq_client = None
        
        #Setting up gemini GenAI Client
        try:
            self.gemini_client = genai.Client(api_key=self.GEMINI_KEY)
        except Exception as e:
            print(f"Error initializing Gemini GenAI client: {e}")
            self.gemini_client = None
    
    def _setup_model(self, trained_model_path):
        self.base_model_name = "Salesforce/codet5-base"
        if os.path.exists(self.trained_model_path):
            print(f"Loading Trained Model from: {self.trained_model_path}")
            self.load_path = self.trained_model_path
        else:
            print(f"Trained model not found (saved_model).")
            print(f"Loading Base Model for testing: {self.base_model_name}")
            self.load_path = self.base_model_name

        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.load_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.load_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def fix(self, buggy_code: str, intention=""):
        # Prefix as used in training
        if intention != "":
            input_text = f"fix intent: {str(intention).strip()} code: {buggy_code}"
        else:
            input_text = "fix: " + buggy_code
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").input_ids
        
        # Generate
        outputs = self.model.generate(inputs, max_length=128)
        
        # Decode
        fixed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fixed_code
    
    def generate_feedback(self, buggy_code: str, fixed_code: str):
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
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"\n[!] Gemini Failed. Switching to Groq...")

        try:
            response = self.groq_client.chat.completions.create(
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
    repairer = Repairer()
    fixed = repairer.fix(code)
    feedback = repairer.generate_feedback(code, fixed)
    print("\n--- FEEDBACK ---")
    print(feedback)
    print("\n--- FIXED CODE ---")
    print(f"Fixed Code >> {fixed}")
