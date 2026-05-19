import os
import sys
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Repairer:
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    groq_client = None
    def __init__(self):
        self._setup_AI_clients()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.trained_model_path = os.path.join(self.script_dir, "training", "t5", "saved_model")
        self._setup_model()
        
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
    
    def _setup_model(self):
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
    
    def _extract_body_snippet(self, code: str) -> tuple[str, str, str]:
        """Return (prefix, snippet, suffix) where snippet is the function body.

        T5 was trained on short body snippets, not full function definitions.
        Sending the full function floods the context with the signature and
        docstring, degrading output quality. We strip those and send only the
        body, then reconstruct the full function around T5's fix.
        """
        import ast as _ast
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return "", code, ""

        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                lines = code.splitlines()
                # body starts after the def line (and any docstring)
                body_start = node.body[0].lineno - 1
                # skip docstring if first statement is a string constant
                if (isinstance(node.body[0], _ast.Expr) and
                        isinstance(node.body[0].value, _ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                    body_start = node.body[1].lineno - 1 if len(node.body) > 1 else body_start
                prefix = "\n".join(lines[:body_start])
                snippet = "\n".join(lines[body_start:])
                return prefix + "\n", snippet, ""

        return "", code, ""

    def fix(self, buggy_code: str) -> str:
        prefix, snippet, _ = self._extract_body_snippet(buggy_code)

        inputs = self.tokenizer(
            snippet,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).input_ids

        outputs = self.model.generate(
            inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

        fixed_snippet = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Reconstruct the full function: original signature + fixed body.
        # T5 sometimes drops leading indentation; restore it from the original
        # body's first non-empty line so the output stays valid Python.
        if prefix and snippet:
            orig_indent = ""
            for line in snippet.splitlines():
                if line.strip():
                    orig_indent = line[: len(line) - len(line.lstrip())]
                    break
            if orig_indent and not fixed_snippet.startswith(orig_indent):
                fixed_snippet = "\n".join(
                    orig_indent + ln if ln.strip() else ln
                    for ln in fixed_snippet.splitlines()
                )
            return prefix + fixed_snippet
        return fixed_snippet
    
    def generate_feedback(self, buggy_code: str, fixed_code: str):
        prompt = f"""
        You are a Code Reviewer.
        Buggy Code: {buggy_code}
        Fixed Code: {fixed_code}
        
        Task:
        1. Analysis: Make a comparison between Buggy Code and Fixed Code. 1 sentence on the bug. Or if no change, say no bug found. Then no need to do reasoning if there's no bug.
        2. Reasoning: 1 or 2 sentences on the fix.
        3. Accuracy: Rate the accuracy of the fix as High, Medium, or Low based on how well it addresses the bug. Be honest and critical in your assessment.
        
        Additional Instructions:
        - Be concise and clear.
        - The suggested fix may be incorrect; act as a reviewer and provide honest feedback on the quality of the fix.
        
        Format:
        Analysis: ...
        Reasoning: ...
        Accuracy: [High/Medium/Low] (based on how well the fix addresses the bug)
        """

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
            return f"[Groq Unavailable] Could not generate feedback: {e}"

if __name__ == "__main__":
    code = "def add(a, b):\n    return a - b"
    repairer = Repairer()
    fixed = repairer.fix(code)
    feedback = repairer.generate_feedback(code, fixed)
    print("\n--- FEEDBACK ---")
    print(feedback)
    print("\n--- FIXED CODE ---")
    print(f"Fixed Code >> {fixed}")
