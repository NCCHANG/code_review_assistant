import os
import sys
import ast

# Import own components
import Predictor
import Repairer

class CodeAssistant:
    functions_and_bugginess = [] # List of tuples: (function_name, is_buggy, confidence)
    functions_fix_feedback = [] # List of tuples: (function_name, fixed_code, feedback)
    def __init__(self):
        self.predictor = Predictor.Predictor()
        self.repairer = Repairer.Repairer()

    def _extract_functions(self, code: str):
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

    def process_file_or_input(self, code_content: str, context=""):
        # ----------------------------------------------------------------
        # EXTRACTING FUNCTIONS FROM RAW CODE
        functions = self._extract_functions(code_content)
        if not functions:
            functions = [("UserSnippet", code_content)]

        print(f"\nAnalyzing {len(functions)} function(s)...\n")
        #--------------------------------------------------------------------
        #PREDICT & REPAIR
        for func_name, func_code in functions:
            print(f"--- Checking: {func_name} ---")
            
            is_buggy, confidence = self.predictor.predict(func_code, threshold=0.30)
            self.functions_and_bugginess.append((func_name, is_buggy, confidence))
            
            if is_buggy:
                print(f"  [STATUS]: BUGGY (Prob: {confidence:.2%})")
                print(f"  [ACTION]: Repairing...")
                try:
                    if context:
                        fixed_code = self.repairer.fix(func_code, intention=context)
                    else:
                        fixed_code = self.repairer.fix(func_code)
                    feedback = self.repairer.generate_feedback(func_code, fixed_code)
                    self.functions_fix_feedback.append((func_name, fixed_code, feedback))
                    print(f"  [FEEDBACK]:\n{feedback}\n")
                    print(f"  [FIX]:\n{fixed_code}\n")
                except Exception as e:
                    print(f"  [ERROR]: Repair failed: {e}")
            else:
                print(f"  [STATUS]: CLEAN (Prob: {confidence:.2%})\n")
    def get_analysis_results(self):
        return self.functions_and_bugginess, self.functions_fix_feedback

def main():
    print("Loading AI Models...")
    code_assistant = CodeAssistant()

    print("\n" + "="*50)
    print("      CODE REVIEW ASSISTANT (Hybrid AI)")
    print("      Supports: Raw Code or File Paths")
    print("="*50)

    while True:
        user_input = input("\nEnter Code or File Path (or 'exit') >> ")
        if user_input.lower() == 'exit':
            break
        
        code_assistant.process_file_or_input(user_input)

if __name__ == "__main__":
    main()
