import subprocess
import json
import os
import tempfile


# Pylint symbol → human-readable title
_SYMBOL_TITLES: dict[str, str] = {
    "missing-return": "Missing Return Statement",
    "inconsistent-return-statements": "Missing Return Statement",
    "expression-not-assigned": "Unused Statement Result",
    "pointless-statement": "Unused Statement Result",
    "line-too-long": "PEP 8 Style",
    "trailing-whitespace": "PEP 8 Style",
    "missing-final-newline": "PEP 8 Style",
    "unused-variable": "Unused Variable",
    "undefined-variable": "Undefined Variable",
    "undefined-loop-variable": "Undefined Variable",
    "bare-except": "Bare Except Clause",
    "broad-exception-caught": "Broad Except Clause",
    "broad-except": "Broad Except Clause",
    "comparison-with-itself": "Comparison Always True/False",
    "singleton-comparison": "Comparison Always True/False",
    "literal-comparison": "Comparison Always True/False",
    "use-implicit-booleaness-not-comparison": "Logic Error",
    "redefined-builtin": "Variable Shadows Builtin",
    "attribute-defined-outside-init": "Attribute Outside __init__",
    "no-member": "Attribute/Method Not Found",
    "invalid-name": "Naming Convention",
}

# Pylint message type → UI severity
_TYPE_SEVERITY: dict[str, str] = {
    "error": "ERROR",
    "fatal": "ERROR",
    "warning": "WARNING",
    "convention": "INFO",
    "refactor": "INFO",
}


class StaticEngine:
    staticResults: str | None = None
    path: str | None = None

    def __init__(self, path: str = ""):
        self.path = path

    # ── File-based analysis (original) ───────────────────────────────────────

    def run(self) -> None:
        """Run pylint on self.path and store the raw JSON output."""
        result = subprocess.run(
            ["pylint", self.path, "--score=no", "-f", "json"],
            capture_output=True,
            text=True,
        )
        self.staticResults = result.stdout

    # ── In-memory code analysis (new) ────────────────────────────────────────

    def run_on_code(self, code: str) -> None:
        """Write *code* to a temp file, run pylint on it, and store results.

        Errors (E) and Warnings (W) are always reported; the PEP-8 line-length
        convention check (C0301) is also enabled. Refactor and other convention
        messages are suppressed to reduce noise on code snippets.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [
                    "pylint", tmp_path,
                    "--score=no",
                    "-f", "json",
                    "--disable=C,R",
                    "--enable=C0301",   # line-too-long
                ],
                capture_output=True,
                text=True,
            )
            self.staticResults = result.stdout or "[]"
            print(f"[StaticEngine] pylint returncode: {result.returncode}")
            print(f"[StaticEngine] pylint stdout: {result.stdout!r}")
            print(f"[StaticEngine] pylint stderr: {result.stderr!r}")
        except FileNotFoundError:
            print("[StaticEngine] ERROR: pylint not found — is it installed?")
            self.staticResults = "[]"
        finally:
            os.unlink(tmp_path)

    # ── Result accessors ─────────────────────────────────────────────────────

    def get_results(self) -> list | None:
        """Return the raw parsed pylint JSON list, or None on error."""
        if self.staticResults is None:
            print("No results available. Run the engine first.")
            return None
        try:
            return json.loads(self.staticResults)
        except json.JSONDecodeError:
            print("Error parsing JSON results.")
            return None

    def get_violations(self) -> list[dict]:
        """Return violations as [{severity, title, description, line_number}].

        Suitable for direct use in the results UI.
        """
        raw = self.get_results()
        if not raw:
            return []

        violations: list[dict] = []
        for item in raw:
            msg_type = item.get("type", "warning").lower()
            severity = _TYPE_SEVERITY.get(msg_type, "INFO")
            symbol = item.get("symbol", "")
            title = _SYMBOL_TITLES.get(symbol) or symbol.replace("-", " ").title()
            violations.append({
                "severity": severity,
                "title": title,
                "description": item.get("message", ""),
                "line_number": item.get("line", 0),
            })
        return violations


if __name__ == "__main__":
    engine = StaticEngine("test_complex_code.py")
    engine.run()
    print(json.dumps(engine.get_violations(), indent=2))
