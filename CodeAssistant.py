import os
import sys
import ast

# Import own components
import Predictor
import Repairer

def extract_functions(code: str):
    """Parses code and returns a list of (function_name, source_code of function (def func ....) ) tuples."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Syntax Error in input code. Cannot parse functions.")
        return []
        
    functions = []
    for node in ast.walk(tree): #walk through whole code tree to find all functions
        if isinstance(node, ast.FunctionDef):
            func_source = ast.get_source_segment(code, node)
            functions.append((node.name, func_source))
    return functions

def process_file_or_input(user_input: str, predictor_model, repairer_model):
    # ----------------------------------------------------------------
    # EXTRACTING FUNCTIONS FROM RAW CODE
    functions = extract_functions(user_input)
    if not functions:
        functions = [("UserSnippet", user_input)]

    print(f"\nAnalyzing {len(functions)} function(s)...\n")
    #--------------------------------------------------------------------
    #PREDICT & REPAIR
    for func_name, func_code in functions:
        print(f"--- Checking: {func_name} ---")
        
        is_buggy, confidence = predictor_model.predict(func_code, threshold=0.30)
        
        if is_buggy:
            print(f"  [STATUS]: BUGGY (Prob: {confidence:.2%})")
            print(f"  [ACTION]: Repairing...")
            try:
                fixed_code = repairer_model.fix(func_code)
                feedback = repairer_model.generate_feedback(func_code, fixed_code)
                print(f"  [FEEDBACK]:\n{feedback}\n")
                print(f"  [FIX]:\n{fixed_code}\n")
            except Exception as e:
                print(f"  [ERROR]: Repair failed: {e}")
        else:
            print(f"  [STATUS]: CLEAN (Prob: {confidence:.2%})\n")

def main():
    print("Loading AI Models...")
    predictor_model = Predictor.Predictor()
    repairer_model = Repairer.Repairer()

    print("\n" + "="*50)
    print("      CODE REVIEW ASSISTANT (Hybrid AI)")
    print("      Supports: Raw Code or File Paths")
    print("="*50)

    while True:
        user_input = input("\nEnter Code or File Path (or 'exit') >> ")
        if user_input.lower() == 'exit':
            break
        
        process_file_or_input(user_input, predictor_model, repairer_model)

if __name__ == "__main__":
    main()
