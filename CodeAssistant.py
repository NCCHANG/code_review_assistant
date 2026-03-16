import os
import sys
import ast

# Import own components
from Predictor import load_model as load_predictor, predict_bug
from real.Repairer import fix_bug

def extract_functions(code):
    """Parses code and returns a list of (function_name, source_code) tuples."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Syntax Error in input code. Cannot parse functions.")
        return []
        
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get the source segment
            func_source = ast.get_source_segment(code, node)
            functions.append((node.name, func_source))
    return functions

def process_file_or_input(user_input, model, vectorizer):
    # Check if input is a file path
    if os.path.exists(user_input):
        try:
            with open(user_input, 'r', encoding='utf-8') as f:
                code_content = f.read()
            print(f"Reading file: {user_input}")
            functions = extract_functions(code_content)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Treat as raw code
        # Try to parse as functions, if failure (e.g. just a snippet), treat as one block
        functions = extract_functions(user_input)
        if not functions:
            functions = [("UserSnippet", user_input)]

    if not functions:
        print("No functions found to analyze.")
        return

    print(f"\nAnalyzing {len(functions)} function(s)...\n")
    
    for func_name, func_code in functions:
        print(f"--- Checking: {func_name} ---")
        
        # 1. PREDICT
        is_buggy, confidence = predict_bug(func_code, model, vectorizer, threshold=0.30)
        
        if is_buggy:
            print(f"  [STATUS]: BUGGY (Prob: {confidence:.2%})")
            print(f"  [ACTION]: Repairing...")
            try:
                fixed_code = fix_bug(func_code)
                print(f"  [FIX]:\n{fixed_code}\n")
            except Exception as e:
                print(f"  [ERROR]: Repair failed: {e}")
        else:
            print(f"  [STATUS]: CLEAN (Prob: {confidence:.2%})\n")

def main():
    print("Loading AI Models...")
    rf_model, rf_vectorizer = load_predictor()
    
    if not rf_model or not rf_vectorizer:
        print("CRITICAL ERROR: Could not load the Random Forest (Predictor) model.")
        return

    print("\n" + "="*50)
    print("      CODE REVIEW ASSISTANT (Hybrid AI)")
    print("      Supports: Raw Code or File Paths")
    print("="*50)

    while True:
        user_input = input("\nEnter Code or File Path (or 'exit') >> ")
        if user_input.lower() == 'exit':
            break
        
        process_file_or_input(user_input, rf_model, rf_vectorizer)

if __name__ == "__main__":
    main()
