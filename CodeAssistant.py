import os
import sys
import ast

import Predictor
import Repairer


class CodeAssistant:
    def __init__(self):
        self.predictor = Predictor.Predictor()
        self.repairer = Repairer.Repairer()
        # Instance-level lists: reset() clears these before each analysis run.
        # Previously these were class attributes, which caused results to
        # accumulate across multiple analyses on the same instance.
        self.functions_and_bugginess = []  # (name, is_buggy, confidence, line_number, source, bug_type)
        self.functions_fix_feedback = []   # (name, fixed_code, feedback)

    def reset(self) -> None:
        """Clear previous analysis results. Called at the start of each run."""
        self.functions_and_bugginess = []
        self.functions_fix_feedback = []

    def _extract_functions(self, code: str) -> list[tuple[str, str, int]]:
        """Parse code and return (function_name, source, start_line) tuples."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print("Syntax Error in input code. Cannot parse functions.")
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_source = ast.get_source_segment(code, node)
                functions.append((node.name, func_source, node.lineno))
        return functions

    def process_file_or_input(self, code_content: str) -> None:
        """Run the full analysis pipeline on *code_content*.

        Results are stored in ``self.functions_and_bugginess`` and
        ``self.functions_fix_feedback`` after calling ``reset()``.
        """
        self.reset()

        functions = self._extract_functions(code_content)
        if not functions:
            functions = [("UserSnippet", code_content, 1)]

        print(f"\nAnalyzing {len(functions)} function(s)...\n")

        for func_name, func_code, line_no in functions:
            print(f"--- Checking: {func_name} ---")

            is_buggy, confidence, bug_type = self.predictor.predict(func_code)
            self.functions_and_bugginess.append(
                (func_name, is_buggy, confidence, line_no, func_code, bug_type)
            )

            if is_buggy:
                print(f"  [STATUS]: BUGGY — {bug_type} (Prob: {confidence:.2%})")
                print(f"  [ACTION]: Repairing...")
                try:
                    fixed_code = self.repairer.fix(func_code)
                    feedback = self.repairer.generate_feedback(func_code, fixed_code)
                    self.functions_fix_feedback.append((func_name, fixed_code, feedback))
                    print(f"  [FEEDBACK]:\n{feedback}\n")
                    print(f"  [FIX]:\n{fixed_code}\n")
                except Exception as e:
                    print(f"  [ERROR]: Repair failed: {e}")
            else:
                print(f"  [STATUS]: CLEAN ({bug_type}, Prob: {confidence:.2%})\n")

    def get_analysis_results(self) -> tuple[list, list]:
        """Return (functions_and_bugginess, functions_fix_feedback)."""
        return self.functions_and_bugginess, self.functions_fix_feedback


def main():
    print("Loading AI Models...")
    code_assistant = CodeAssistant()

    print("\n" + "=" * 50)
    print("      CODE REVIEW ASSISTANT (Hybrid AI)")
    print("      Supports: Raw Code or File Paths")
    print("=" * 50)

    while True:
        user_input = input("\nEnter Code or File Path (or 'exit') >> ").strip()
        if user_input.lower() == "exit":
            break
        if os.path.isfile(user_input):
            with open(user_input, "r", encoding="utf-8") as f:
                code_content = f.read()
            print(f"Reading file: {user_input}")
        else:
            code_content = user_input
        code_assistant.process_file_or_input(code_content)


if __name__ == "__main__":
    main()
