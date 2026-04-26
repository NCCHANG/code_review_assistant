import re
import pandas as pd

# ── Taxonomy ──────────────────────────────────────────────────────────────────
# Priority order matters: first match wins.
TAXONOMY = [
    {
        "label": "security",
        "msg_keywords": ["cve", "vulnerabil", "injection", "xss", "csrf",
                         "sanitiz", "escape", "auth", "permission", "privilege",
                         "dos", "denial", "exploit", "exposure", "secret",
                         "sensitive", "disclose"],
        "diff_patterns": [r"cve", r"escape\(", r"sanitize"],
    },
    {
        "label": "null_reference",
        "msg_keywords": ["nonetype", "attributeerror", "null", "none check",
                         "missing attribute", "not none", "is none", "truthy",
                         "falsy"],
        "diff_patterns": [r"if \w+ is None", r"if \w+ is not None",
                          r"AttributeError", r"getattr\("],
    },
    {
        "label": "exception_handling",
        "msg_keywords": ["exception", "crash", "traceback", "unhandled",
                         "raises", "keyerror", "valueerror", "typeerror",
                         "indexerror", "runtimeerror", "oserror", "prevented",
                         "prevent"],
        "diff_patterns": [r"\btry\b", r"\bexcept\b", r"\braise\b",
                          r"except \w+Error"],
    },
    {
        "label": "logic_error",
        "msg_keywords": ["wrong", "incorrect", "off-by-one", "condition",
                         "operator", "comparison", "invalid", "unexpected",
                         "ordering", "precedence", "replaced", "misplaced",
                         "queryset", "filter", "formset", "inline", "lookup",
                         "behavior", "behaviour", "incorrectly", "return value"],
        "diff_patterns": [r"[><=!]=", r"\bnot\b.*\bin\b", r"\band\b.*\bor\b"],
    },
    {
        "label": "data_handling",
        "msg_keywords": ["serial", "deserial", "pars", "encod", "decod",
                         "format", "render", "templat", "json", "xml", "csv",
                         "response", "request", "header", "content-type",
                         "mimetype", "charset"],
        "diff_patterns": [r"json\.", r"\.loads\(", r"\.dumps\(",
                          r"Content-Type", r"charset"],
    },
    {
        "label": "type_error",
        "msg_keywords": ["type", "cast", "convert", "string", "integer",
                         "bytes", "unicode", "isinstance", "coerce"],
        "diff_patterns": [r"\bisinstance\(", r"\.encode\(", r"\.decode\(",
                          r"\bstr\(", r"\bint\(", r"\bbytes\("],
    },
    {
        "label": "null_reference",
        "msg_keywords": [],
        "diff_patterns": [r"\bNone\b"],
    },
    {
        "label": "edge_case",
        "msg_keywords": ["edge", "corner", "boundary", "empty", "zero",
                         "negative", "when no", "if no", "strict", "validation",
                         "validate", "check", "missing value", "not provided"],
        "diff_patterns": [r"if not \w+", r"len\(\w+\) == 0", r"== \[\]",
                          r"== \{\}"],
    },
    {
        "label": "compatibility",
        "msg_keywords": ["deprecat", "compat", "python 3", "python3",
                         "migration", "upgrade", "backport", "legacy",
                         "support for", "dropped support", "removed support"],
        "diff_patterns": [r"sys\.version", r"PY2", r"six\."],
    },
]

FALLBACK_LABEL = "other"


def classify(commit_msg: str, buggy_hunk: str, fixed_hunk: str) -> str:
    msg = commit_msg.lower()
    diff_context = (buggy_hunk + "\n" + fixed_hunk).lower()

    seen = set()
    for rule in TAXONOMY:
        label = rule["label"]
        if label in seen:
            continue

        if any(kw in msg for kw in rule["msg_keywords"]):
            return label

        if any(re.search(p, diff_context) for p in rule["diff_patterns"]):
            seen.add(label)
            return label

    return FALLBACK_LABEL


def label_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} pairs from {input_csv}")

    df["bug_type"] = df.apply(
        lambda row: classify(
            str(row["commit_message"]),
            str(row["input_text"]),
            str(row["target_text"]),
        ),
        axis=1,
    )

    df.to_csv(output_csv, index=False)
    print(f"Saved labelled dataset to {output_csv}\n")

    print("── Bug type distribution ──────────────────────")
    counts = df["bug_type"].value_counts()
    total = len(df)
    for label, count in counts.items():
        bar = "█" * (count * 40 // total)
        print(f"  {label:<22} {count:>5}  ({count/total*100:.1f}%)  {bar}")
    print(f"  {'TOTAL':<22} {total:>5}")


if __name__ == "__main__":
    label_dataset("massive_local_bugs_dataset.csv", "massive_local_bugs_dataset_labelled.csv")
